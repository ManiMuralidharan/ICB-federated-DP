# =============================================================================
# crpr_sensitivity.py
#
# CR/PR-only endpoint sensitivity analysis (reviewer request).
#
# The primary analysis defines responders as CR, PR or SD (clinical benefit)
# versus PD. This script re-labels patients under a STRICTER endpoint,
# responders = CR or PR, non-responders = SD or PD, by recovering the granular
# RECIST category for every patient from each cohort's ORIGINAL publication
# supplement, and then re-runs the leave-one-cohort-out centralized reference
# under both endpoints to show whether the conclusions are robust.
#
# It writes CR_PR_Sensitivity.csv with BOTH the per-cohort mean LOCO AUROC
# (the headline value, matching the main-text convention of averaging the five
# held-out-cohort AUROCs) AND the pooled LOCO AUROC, for each feature set under
# the primary and strict endpoints. It also writes CR_PR_reclassification.csv
# with per-cohort counts. The original RECIST category is retained for every
# patient so that each stable-disease reassignment is auditable (the script
# verifies that every reclassified patient had an original category of SD).
#
# ---------------------------------------------------------------------------
# DATA YOU MUST SUPPLY (not redistributed in this repository; obtain from the
# original publications and place in $ICB_CLINICAL_DIR):
#
#   Cohort      RECIST source file                                  Column
#   Hugo        Hugo et al, Cell 2016, mmc1.xls (sheet S1A)         irRECIST
#   Riaz        Riaz et al, Cell 2017, Table S2 (response.xlsx)     Response
#   Gide        Gide et al, Cancer Cell 2019, Gide_clinical.tsv     response
#   Liu         Liu et al, Nat Med 2019, MOESM4 (Suppl. Table 1)    BR
#   IMvigor210  Mariathasan et al, Nature 2018, IMvigor_clinical    Best Confirmed Overall Response
#
# Patient identifiers are matched case-insensitively to the processed
# {Cohort}_clean.csv files in $ICB_DATA_DIR.
#
# ENVIRONMENT:
#   ICB_DATA_DIR      folder with processed {Cohort}_clean.csv (gene + response_binary)
#   ICB_CLINICAL_DIR  folder with the five RECIST source files named below
#
# Run:  ICB_DATA_DIR=./processed ICB_CLINICAL_DIR=./clinical python crpr_sensitivity.py
# Deps: numpy pandas scikit-learn openpyxl xlrd
# =============================================================================

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score

DATA_DIR = os.environ.get("ICB_DATA_DIR", "./processed")
CLIN_DIR = os.environ.get("ICB_CLINICAL_DIR", "./clinical")

# file names of the RECIST source supplements, inside CLIN_DIR
CLIN_FILES = {
    "Hugo":       "Hugo_mmc1.xls",            # Cell 2016 supplement, sheet S1A
    "Riaz":       "Riaz_response.xlsx",       # Cell 2017 Table S2
    "Gide":       "Gide_clinical.tsv",
    "Liu":        "Liu_MOESM4.xlsx",          # Nat Med 2019 Supplemental Table 1
    "IMvigor210": "IMvigor_clinical.csv",
}

TRAFFICKING_GENES = ["CD274", "CMTM6", "CMTM4", "DRG2", "RAB11A", "RAB8A",
                     "SNX1", "SNX2", "SNX27", "VPS35"]
IFN_GAMMA_GENES = ["IFNG", "STAT1", "CXCL9", "CXCL10", "IDO1", "HLA-DRA"]
FEATURE_SETS = {
    "Trafficking": TRAFFICKING_GENES,
    "IFN_gamma":   IFN_GAMMA_GENES,
    "Combined":    TRAFFICKING_GENES + IFN_GAMMA_GENES,
}
COHORTS = ["Hugo", "Riaz", "Gide", "Liu", "IMvigor210"]


def _norm(s):
    return str(s).strip().upper()


def _strict_label(recist):
    """Strict endpoint: CR/PR -> 1 (responder); SD/PD -> 0 (non-responder).

    Minor response (MR) and not-evaluable (NE) are NOT equivalent to stable or
    progressive disease, so they are returned as NaN and EXCLUDED from this
    analysis rather than forced into the non-responder group. The analysis
    cohorts used here contain only CR/PR/SD/PD (any MR/NE were removed during
    the original preprocessing), so in practice no patients are excluded; the
    exclusion is reported so the behavior is explicit and verifiable.
    """
    r = str(recist).strip().upper()
    if r in ("CR", "PR", "COMPLETE RESPONSE", "PARTIAL RESPONSE"):
        return 1
    if r in ("SD", "PD", "STABLE DISEASE", "PROGRESSIVE DISEASE"):
        return 0
    return np.nan   # MR, NE, or any other / missing category -> excluded


def _canon_recist(recist):
    """Canonical RECIST bucket: CR, PR, SD, PD, or OTHER (MR/NE/missing).

    Retained alongside the strict label so that every reclassified patient can
    be audited back to a specific original category rather than inferred.
    """
    r = str(recist).strip().upper()
    if r in ("CR", "COMPLETE RESPONSE"):   return "CR"
    if r in ("PR", "PARTIAL RESPONSE"):    return "PR"
    if r in ("SD", "STABLE DISEASE"):      return "SD"
    if r in ("PD", "PROGRESSIVE DISEASE"): return "PD"
    return "OTHER"


_STRICT_FROM_CANON = {"CR": 1, "PR": 1, "SD": 0, "PD": 0}  # OTHER -> excluded


def _load_recist(cohort):
    """Return {normalized_patient_id: canonical_RECIST_category} from source."""
    path = os.path.join(CLIN_DIR, CLIN_FILES[cohort])
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing RECIST source for {cohort}: {path}. "
            f"Obtain it from the original publication (see header) and place it in "
            f"ICB_CLINICAL_DIR.")
    if cohort == "Hugo":
        d = pd.read_excel(path, sheet_name="S1A", header=2)
        key, col = "Patient ID", "irRECIST"
    elif cohort == "Riaz":
        d = pd.read_excel(path, header=2)
        key, col = "Patient", "Response"
    elif cohort == "Gide":
        d = pd.read_csv(path, sep="\t")
        key, col = "sample_id", "response"
    elif cohort == "Liu":
        d = pd.read_excel(path, sheet_name="Supplemental Table 1", header=2)
        key, col = d.columns[0], "BR"
    elif cohort == "IMvigor210":
        d = pd.read_csv(path)
        key, col = "Unnamed: 0", "Best Confirmed Overall Response"
    else:
        raise KeyError(cohort)
    return {_norm(p): _canon_recist(r) for p, r in zip(d[key], d[col])}


def _load_clean(cohort):
    df = pd.read_csv(os.path.join(DATA_DIR, f"{cohort}_clean.csv"))
    if "response_binary" not in df.columns:
        raise ValueError(f"{cohort}_clean.csv missing response_binary")
    return df


def _loco_auroc(data, label_col, genes):
    """Leave-one-cohort-out AUROC (centralized L2 logistic reference).

    Standardization is TRAIN-DERIVED: the mean and standard deviation are
    estimated on the pooled training cohorts only and applied to the held-out
    cohort, so the held-out cohort's own distribution is not used (no
    transductive normalization).

    Returns the mean of the per-cohort held-out AUROCs (the same aggregation
    convention as the main text), which is the headline value, and the pooled
    AUROC as a secondary value.
    """
    cohorts = list(data)
    per_cohort = []
    Y, P = [], []
    for held in cohorts:
        train = [c for c in cohorts if c != held]
        Xtr_raw = np.vstack([data[c][genes].to_numpy(float) for c in train])
        mean = np.nanmean(Xtr_raw, axis=0)
        std = np.nanstd(Xtr_raw, axis=0) + 1e-8
        Xtr = np.nan_to_num((Xtr_raw - mean) / std)
        ytr = np.concatenate([data[c][label_col].astype(int).values for c in train])
        if len(np.unique(ytr)) < 2:
            continue
        clf = LogisticRegressionCV(Cs=10, cv=5, penalty="l2",
                                   max_iter=1000, scoring="roc_auc").fit(Xtr, ytr)
        Xte = np.nan_to_num((data[held][genes].to_numpy(float) - mean) / std)
        yte = data[held][label_col].astype(int).values
        pte = clf.predict_proba(Xte)[:, 1]
        Y.append(yte); P.append(pte)
        if len(np.unique(yte)) >= 2:
            per_cohort.append(roc_auc_score(yte, pte))
    mean_auroc = float(np.mean(per_cohort)) if per_cohort else np.nan
    pooled_auroc = roc_auc_score(np.concatenate(Y), np.concatenate(P))
    return mean_auroc, pooled_auroc


def main():
    # attach strict labels and report reclassification
    data = {}
    rows = []
    total_flipped = 0
    total_flipped_sd = 0
    for c in COHORTS:
        df = _load_clean(c)
        rec = _load_recist(c)                       # id -> canonical category
        ids = df.iloc[:, 0].map(_norm)
        raw = ids.map(rec)                           # CR/PR/SD/PD/OTHER or NaN
        matched = int(raw.notna().sum())
        strict = raw.map(_STRICT_FROM_CANON)         # OTHER/unmatched -> NaN
        df = df.assign(recist_raw=raw.values, strict=strict.values)
        prim = int(df["response_binary"].sum())
        sresp = int(np.nansum(strict.values))
        # reclassified = primary responder (CR/PR/SD) now a strict non-responder
        flip_mask = (df["response_binary"] == 1) & (df["strict"] == 0)
        flipped = int(flip_mask.sum())
        # EXPLICIT audit: every reclassified patient's ORIGINAL category must be SD
        flipped_sd = int((flip_mask & (df["recist_raw"] == "SD")).sum())
        flipped_not_sd = flipped - flipped_sd
        total_flipped += flipped
        total_flipped_sd += flipped_sd
        rows.append({"Cohort": c, "n": len(df), "Primary_responders": prim,
                     "Strict_responders": sresp, "SD_reclassified": flipped,
                     "SD_verified": flipped_sd, "Non_SD_reclassified": flipped_not_sd,
                     "RECIST_matched": f"{matched}/{len(df)}"})
        data[c] = df.dropna(subset=["strict"])
        if matched < len(df):
            print(f"  WARNING: {c} matched only {matched}/{len(df)} patients to RECIST")
        if flipped_not_sd > 0:
            print(f"  WARNING: {c} reclassified {flipped_not_sd} patient(s) whose "
                  f"original RECIST category was NOT stable disease")

    counts = pd.DataFrame(rows)
    print("\nReclassification under the strict CR/PR endpoint:")
    print(counts.to_string(index=False))
    print(f"\nSD audit: {total_flipped_sd}/{total_flipped} reclassified patients "
          f"had an original RECIST category of SD "
          f"({'PASS' if total_flipped_sd == total_flipped else 'CHECK FAILED'}).")

    # AUROC under both endpoints (mean of per-cohort AUROCs; pooled as secondary)
    out = []
    print("\nLOCO AUROC (centralized reference, train-derived standardization):")
    print(f"{'Feature set':14s} {'Primary mean':>13s} {'Strict mean':>12s}"
          f" {'Primary pooled':>15s} {'Strict pooled':>14s}")
    for name, genes in FEATURE_SETS.items():
        pm, pp = _loco_auroc(data, "response_binary", genes)
        sm, sp = _loco_auroc(data, "strict", genes)
        out.append({"Feature_set": name,
                    "AUROC_primary_mean": round(pm, 3), "AUROC_strict_mean": round(sm, 3),
                    "AUROC_primary_pooled": round(pp, 3), "AUROC_strict_pooled": round(sp, 3)})
        print(f"{name:14s} {pm:>13.3f} {sm:>12.3f} {pp:>15.3f} {sp:>14.3f}")

    res = pd.DataFrame(out)
    counts.to_csv("CR_PR_reclassification.csv", index=False)
    res.to_csv("CR_PR_Sensitivity.csv", index=False)
    print("\nWrote CR_PR_reclassification.csv and CR_PR_Sensitivity.csv")


if __name__ == "__main__":
    main()
