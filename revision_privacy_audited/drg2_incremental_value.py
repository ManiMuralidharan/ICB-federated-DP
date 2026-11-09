# =============================================================================
# drg2_incremental_value.py
#
# Incremental-value test: does adding DRG2 to the IFN-gamma signature improve
# leave-one-cohort-out (LOCO) prediction of ICB response?
#
# Compares:
#     Base  = IFN-gamma module  (IFNG, STAT1, CXCL9, CXCL10, IDO1, HLA-DRA)
#     +DRG2 = IFN-gamma module + DRG2
#
# Protocol mirrors the paper's centralized reference:
#   - local (per-cohort) z-score normalization of each feature
#   - LogisticRegressionCV(Cs=10, cv=5, penalty='l2', max_iter=1000)
#   - leave-one-cohort-out: fit on all-but-one, predict held-out cohort
#   - pool held-out predictions across folds, then compare the two models on
#     the SAME pooled patients with a PAIRED bootstrap
#
# Reports dAUROC, dAUPRC, dBrier (point estimate + 95% CI + P(+DRG2 better)),
# for the full cohort set and melanoma-only.
#
# Run:  python drg2_incremental_value.py
# Deps: numpy pandas scikit-learn   (pip install numpy pandas scikit-learn)
# =============================================================================

import warnings
warnings.filterwarnings("ignore")   # silence sklearn version FutureWarnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

# ----------------------------------------------------------------------------
# CONFIG  -- edit COHORTS to match the clean CSVs present in this folder
# ----------------------------------------------------------------------------
ALL_COHORTS      = ["Hugo", "Riaz", "Gide", "Liu", "IMvigor210"]
MELANOMA_COHORTS = ["Hugo", "Riaz", "Gide", "Liu"]

IFN_GENES = ["IFNG", "STAT1", "CXCL9", "CXCL10", "IDO1", "HLA-DRA"]
BASE_SET  = IFN_GENES
DRG2_SET  = IFN_GENES + ["DRG2"]

N_BOOT = 2000
SEED   = 42


def load_and_normalize_cohort(cohort, genes):
    """Load {cohort}_clean.csv, select genes, local z-score. Returns (X, y)."""
    df = pd.read_csv(f"{cohort}_clean.csv")
    missing = [g for g in genes if g not in df.columns]
    if missing:
        raise KeyError(f"{cohort}: missing columns {missing}")
    X_raw = df[genes].to_numpy(dtype=float)
    y     = df["response_binary"].to_numpy(dtype=int)
    mean  = np.nanmean(X_raw, axis=0)
    std   = np.nanstd(X_raw, axis=0) + 1e-8
    X     = (X_raw - mean) / std
    X     = np.nan_to_num(X, nan=0.0)      # impute to 0 after z-scoring
    return X, y


def loco_pooled_predictions(cohorts, genes):
    """Leave-one-cohort-out. Returns pooled y_true and y_prob aligned by patient,
    in a fixed cohort/patient order so Base and +DRG2 are paired."""
    data = {c: load_and_normalize_cohort(c, genes) for c in cohorts}
    y_true_all, y_prob_all = [], []
    for held in cohorts:
        X_tr = np.vstack([data[c][0] for c in cohorts if c != held])
        y_tr = np.concatenate([data[c][1] for c in cohorts if c != held])
        X_te, y_te = data[held]
        clf = LogisticRegressionCV(Cs=10, cv=5, penalty="l2",
                                   max_iter=1000, scoring="roc_auc")
        clf.fit(X_tr, y_tr)
        p_te = clf.predict_proba(X_te)[:, 1]
        y_true_all.append(y_te)
        y_prob_all.append(p_te)
    return np.concatenate(y_true_all), np.concatenate(y_prob_all)


def paired_bootstrap(y, p_base, p_drg2, n_boot=N_BOOT, seed=SEED):
    """Paired bootstrap of metric differences (+DRG2 minus Base)."""
    rng = np.random.default_rng(seed)
    n = len(y)

    def metrics(yy, pp):
        return (roc_auc_score(yy, pp),
                average_precision_score(yy, pp),
                brier_score_loss(yy, pp))

    auc_b, ap_b, br_b = metrics(y, p_base)
    auc_d, ap_d, br_d = metrics(y, p_drg2)
    point = {"dAUROC": auc_d - auc_b,
             "dAUPRC": ap_d - ap_b,
             "dBrier": br_d - br_b}   # Brier: negative = +DRG2 better

    dauroc, dauprc, dbrier = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue                              # skip degenerate resample
        a_b, p_b, b_b = metrics(yy, p_base[idx])
        a_d, p_d, b_d = metrics(yy, p_drg2[idx])
        dauroc.append(a_d - a_b)
        dauprc.append(p_d - p_b)
        dbrier.append(b_d - b_b)

    def summarize(arr, better="higher"):
        arr = np.array(arr)
        lo, hi = np.percentile(arr, [2.5, 97.5])
        frac = np.mean(arr > 0) if better == "higher" else np.mean(arr < 0)
        return lo, hi, frac

    return point, {
        "dAUROC": summarize(dauroc, "higher"),
        "dAUPRC": summarize(dauprc, "higher"),
        "dBrier": summarize(dbrier, "lower"),   # lower Brier is better
    }


def run(label, cohorts):
    print(f"\n{'='*70}\n{label}  (cohorts: {', '.join(cohorts)})\n{'='*70}")
    y_b, p_b = loco_pooled_predictions(cohorts, BASE_SET)
    y_d, p_d = loco_pooled_predictions(cohorts, DRG2_SET)
    assert np.array_equal(y_b, y_d), "patient order mismatch between models"
    y = y_b

    auc_b = roc_auc_score(y, p_b); auc_d = roc_auc_score(y, p_d)
    ap_b  = average_precision_score(y, p_b); ap_d = average_precision_score(y, p_d)
    br_b  = brier_score_loss(y, p_b); br_d = brier_score_loss(y, p_d)
    print(f"  n = {len(y)} pooled held-out patients "
          f"({int(y.sum())} responders, prevalence {y.mean():.2f})")
    print(f"  {'metric':8s}  {'IFN-gamma':>12s}  {'IFN+DRG2':>12s}")
    print(f"  {'AUROC':8s}  {auc_b:12.3f}  {auc_d:12.3f}")
    print(f"  {'AUPRC':8s}  {ap_b:12.3f}  {ap_d:12.3f}")
    print(f"  {'Brier':8s}  {br_b:12.3f}  {br_d:12.3f}")

    point, ci = paired_bootstrap(y, p_b, p_d)
    print(f"\n  Paired bootstrap ({N_BOOT} reps), delta = (IFN+DRG2) - (IFN-gamma):")
    for m in ["dAUROC", "dAUPRC", "dBrier"]:
        lo, hi, frac = ci[m]
        direction = "P(+DRG2 better)"
        print(f"    {m}: {point[m]:+.4f}  [95% CI {lo:+.4f}, {hi:+.4f}]  "
              f"{direction} = {frac:.2f}")
    return point, ci


if __name__ == "__main__":
    import os
    present = [c for c in ALL_COHORTS if os.path.exists(f"{c}_clean.csv")]
    melanoma = [c for c in MELANOMA_COHORTS if os.path.exists(f"{c}_clean.csv")]

    if len(present) == len(ALL_COHORTS):
        run("FULL 5-COHORT SET", present)
    else:
        missing = set(ALL_COHORTS) - set(present)
        print(f"** {', '.join(sorted(missing))}_clean.csv not found — "
              f"skipping full-set run. **")

    run("MELANOMA-ONLY", melanoma)
