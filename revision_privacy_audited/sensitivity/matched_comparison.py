# =============================================================================
# matched_comparison.py
#
# Matched centralized-vs-federated comparison (Reviewer 1, Major 4;
# Reviewer 3, Major 4).
#
# THE CONFOUND:
#   The published centralized reference tunes L2 regularization by 5-fold
#   cross-validation (LogisticRegressionCV, a closed-form-ish sklearn solver),
#   while the federated model is a single-layer torch logistic regression
#   (nn.Linear) trained with Adam under fixed hyperparameters. A raw
#   centralized-vs-federated gap therefore mixes THREE things:
#     (i)   the estimator / optimizer (sklearn CV-tuned vs torch Adam, fixed),
#     (ii)  cross-validated vs fixed regularization, and
#     (iii) federation (data partitioning + FedAvg averaging) and privacy.
#
# WHAT THIS SCRIPT DOES:
#   On identical leave-one-cohort-out folds, identical features, and the same
#   train-derived standardization, it fits TWO centralized models on the POOLED
#   training cohorts:
#     A. centralized_cv      : LogisticRegressionCV  (the published reference)
#     B. centralized_matched : a torch nn.Linear trained with the FEDERATED
#                              model's EXACT optimizer and hyperparameters
#                              (Adam, lr, weight_decay, batch_size, and the same
#                              total number of passes = epochs * rounds), pooled
#                              and WITHOUT DP noise.
#   Model B differs from the federated model in ONE respect only: the data are
#   pooled rather than partitioned-and-averaged (and no DP noise). So:
#       Delta_CV_minus_matched  = estimator + hyperparameter-selection effect
#       Delta_matched_minus_fed = pure federation (+ privacy) effect
#   If an epsilon=inf federated AUROC is supplied (ICB_RESULTS_CSV), it is shown
#   alongside so the matched federation effect can be read directly.
#
# ---------------------------------------------------------------------------
# DEFAULTS BELOW ARE YOUR pipeline.py CONFIG (revised, privacy-audited run):
#   Adam, lr=0.005, weight_decay=1e-4, batch_size=8, epochs=5, rounds=15.
#   nn.Linear(input_dim, 1), BCEWithLogitsLoss. Model B uses NO DP noise and NO
#   gradient clipping (it is the non-private centralized matched model); the
#   federated epsilon=inf column is the non-private federated counterpart.
#   If you change CONFIG in pipeline.py, set the env vars to match, or Model B
#   is not actually matched.
#
# ENVIRONMENT:
#   ICB_DATA_DIR     folder with processed {Cohort}_clean.csv (gene + response_binary)
#   FED_LR           Adam learning rate                   (default 0.005)
#   FED_WD           Adam weight_decay (L2)               (default 1e-4)
#   FED_BATCH        batch size                           (default 8)
#   FED_EPOCHS       local epochs per round               (default 5)
#   FED_ROUNDS       federated rounds                     (default 15)
#   FED_SEED         torch seed                           (default 42)
#   ICB_RESULTS_CSV  optional: main-results CSV with federated AUROC at eps=inf
#                    (e.g. results/Comprehensive_Results.csv)
#
# Run:  ICB_DATA_DIR=./processed \
#       ICB_RESULTS_CSV=./results/Comprehensive_Results.csv \
#       python matched_comparison.py
#
# Writes Matched_Comparison.csv.
# Deps: numpy pandas scikit-learn torch
# =============================================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn

DATA_DIR = os.environ.get("ICB_DATA_DIR", "./processed")
FED_LR = float(os.environ.get("FED_LR", "0.005"))
FED_WD = float(os.environ.get("FED_WD", "1e-4"))
FED_BATCH = int(os.environ.get("FED_BATCH", "8"))
FED_EPOCHS = int(os.environ.get("FED_EPOCHS", "5"))
FED_ROUNDS = int(os.environ.get("FED_ROUNDS", "15"))
FED_SEED = int(os.environ.get("FED_SEED", "42"))
RESULTS_CSV = os.environ.get("ICB_RESULTS_CSV", "")

TRAFFICKING_GENES = ["CD274", "CMTM6", "CMTM4", "DRG2", "RAB11A", "RAB8A",
                     "SNX1", "SNX2", "SNX27", "VPS35"]
IFN_GAMMA_GENES = ["IFNG", "STAT1", "CXCL9", "CXCL10", "IDO1", "HLA-DRA"]
FEATURE_SETS = {
    "Trafficking": TRAFFICKING_GENES,
    "IFN_gamma":   IFN_GAMMA_GENES,
    "Combined":    TRAFFICKING_GENES + IFN_GAMMA_GENES,
}
COHORTS = ["Hugo", "Riaz", "Gide", "Liu", "IMvigor210"]


def _load_clean(cohort):
    df = pd.read_csv(os.path.join(DATA_DIR, f"{cohort}_clean.csv"))
    if "response_binary" not in df.columns:
        raise ValueError(f"{cohort}_clean.csv missing response_binary")
    return df


def _standardize(Xtr_raw, Xte_raw):
    """Train-derived standardization: fit on pooled train, apply to held-out."""
    mean = np.nanmean(Xtr_raw, axis=0)
    std = np.nanstd(Xtr_raw, axis=0) + 1e-8
    Xtr = np.nan_to_num((Xtr_raw - mean) / std)
    Xte = np.nan_to_num((Xte_raw - mean) / std)
    return Xtr, Xte


def _fit_torch_matched(Xtr, ytr):
    """nn.Linear logistic regression trained EXACTLY like the federated client
    (Adam, same lr/weight_decay/batch_size, total passes = epochs*rounds),
    but on pooled data and without DP noise."""
    torch.manual_seed(FED_SEED)
    np.random.seed(FED_SEED)
    model = nn.Linear(Xtr.shape[1], 1)
    opt = torch.optim.Adam(model.parameters(), lr=FED_LR, weight_decay=FED_WD)
    lossf = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32).view(-1, 1)
    n = len(Xt)
    total_passes = FED_EPOCHS * FED_ROUNDS       # match federated total exposure
    model.train()
    for _ in range(total_passes):
        perm = torch.randperm(n)
        for i in range(0, n, FED_BATCH):
            idx = perm[i:i + FED_BATCH]
            opt.zero_grad()
            loss = lossf(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
    model.eval()
    return model


def _predict_torch(model, Xte):
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(Xte, dtype=torch.float32))).view(-1).numpy()


def _loco(data, genes, kind):
    """Leave-one-cohort-out AUROC; returns (per-cohort mean, pooled)."""
    cohorts = list(data)
    per_cohort, Y, P = [], [], []
    for held in cohorts:
        train = [c for c in cohorts if c != held]
        Xtr_raw = np.vstack([data[c][genes].to_numpy(float) for c in train])
        ytr = np.concatenate([data[c]["response_binary"].astype(int).values
                              for c in train])
        Xte_raw = data[held][genes].to_numpy(float)
        yte = data[held]["response_binary"].astype(int).values
        if len(np.unique(ytr)) < 2:
            continue
        Xtr, Xte = _standardize(Xtr_raw, Xte_raw)
        if kind == "cv":
            clf = LogisticRegressionCV(Cs=10, cv=5, penalty="l2",
                                       max_iter=1000, scoring="roc_auc").fit(Xtr, ytr)
            pte = clf.predict_proba(Xte)[:, 1]
        elif kind == "matched":
            model = _fit_torch_matched(Xtr, ytr)
            pte = _predict_torch(model, Xte)
        else:
            raise KeyError(kind)
        Y.append(yte); P.append(pte)
        if len(np.unique(yte)) >= 2:
            per_cohort.append(roc_auc_score(yte, pte))
    mean_auroc = float(np.mean(per_cohort)) if per_cohort else np.nan
    pooled = roc_auc_score(np.concatenate(Y), np.concatenate(P)) if Y else np.nan
    return mean_auroc, pooled


def _federated_auroc(feature_set):
    """Optional: federated non-private (eps=inf) mean AUROC from the main CSV."""
    if not RESULTS_CSV or not os.path.exists(RESULTS_CSV):
        return np.nan
    try:
        r = pd.read_csv(RESULTS_CSV)
    except Exception:
        return np.nan
    cols = {c.lower(): c for c in r.columns}
    fcol = next((cols[k] for k in cols if "feature" in k), None) or \
           next((cols[k] for k in cols if "model" in k), None)
    acol = next((cols[k] for k in cols if k == "auroc" or "auroc" in k or "auc" in k), None)
    ecol = next((cols[k] for k in cols if "eps" in k or "budget" in k or "privacy" in k), None)
    mcol = cols.get("model") or next((cols[k] for k in cols
                                      if "method" in k or "algo" in k or "strategy" in k), None)
    if mcol == fcol:
        mcol = None
    if fcol is None or acol is None:
        return np.nan
    sub = r.copy()
    if ecol is not None:
        s = sub[ecol].astype(str).str.lower()
        sub = sub[s.str.contains("inf") | (pd.to_numeric(sub[ecol], errors="coerce") == np.inf)]
    if mcol is not None:
        mm = sub[mcol].astype(str).str.lower()
        fed = sub[mm.str.contains("fedavg")]
        if len(fed) == 0:
            fed = sub[mm.str.contains("fed") | mm.str.contains(r"\bfl")]
        sub = fed
    key = feature_set.replace("_", "").lower()
    hit = sub[sub[fcol].astype(str).str.replace("_", "").str.lower().str.contains(key)]
    if len(hit) == 0:
        return np.nan
    return float(pd.to_numeric(hit[acol], errors="coerce").dropna().mean())


def main():
    data = {c: _load_clean(c) for c in COHORTS}
    print(f"Federated-matched config: Adam lr={FED_LR}, weight_decay={FED_WD}, "
          f"batch_size={FED_BATCH}, passes={FED_EPOCHS}x{FED_ROUNDS}="
          f"{FED_EPOCHS*FED_ROUNDS}, seed={FED_SEED}")
    print("(Confirm these equal pipeline.py's CONFIG.)\n")

    rows = []
    print(f"{'Feature set':14s} {'cent_CV':>8s} {'cent_matched':>13s} {'fed(eps=inf)':>13s}"
          f" {'CV-matched':>11s} {'matched-fed':>12s}")
    for name, genes in FEATURE_SETS.items():
        cv_m, cv_p = _loco(data, genes, "cv")
        mt_m, mt_p = _loco(data, genes, "matched")
        fed_m = _federated_auroc(name)
        d_est = cv_m - mt_m
        d_fed = (mt_m - fed_m) if not np.isnan(fed_m) else np.nan
        rows.append({
            "Feature_set": name,
            "Centralized_CV_mean": round(cv_m, 3), "Centralized_CV_pooled": round(cv_p, 3),
            "Centralized_matched_mean": round(mt_m, 3), "Centralized_matched_pooled": round(mt_p, 3),
            "Federated_epsinf_mean": (round(fed_m, 3) if not np.isnan(fed_m) else "NA"),
            "Delta_CV_minus_matched": round(d_est, 3),
            "Delta_matched_minus_federated": (round(d_fed, 3) if not np.isnan(d_fed) else "NA"),
        })
        fed_s = f"{fed_m:13.3f}" if not np.isnan(fed_m) else f"{'NA':>13s}"
        dfed_s = f"{d_fed:12.3f}" if not np.isnan(d_fed) else f"{'NA':>12s}"
        print(f"{name:14s} {cv_m:8.3f} {mt_m:13.3f} {fed_s} {d_est:11.3f} {dfed_s}")

    pd.DataFrame(rows).to_csv("Matched_Comparison.csv", index=False)
    print("\nWrote Matched_Comparison.csv")
    print("\nInterpretation:")
    print("  Delta_CV_minus_matched      = estimator + hyperparameter-selection effect")
    print("                                (sklearn CV-tuned vs torch fixed, pooled, same folds).")
    print("  Delta_matched_minus_federated = pure federation (+ privacy) effect, with the")
    print("                                optimizer and hyperparameters held identical.")
    print("  A near-zero Delta_CV_minus_matched means the centralized-vs-federated gap is")
    print("  not driven by hyperparameter selection or estimator choice.")


if __name__ == "__main__":
    main()
