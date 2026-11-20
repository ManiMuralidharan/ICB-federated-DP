"""S-NORM rerun: non-private FedProx (epsilon = inf) under leave-one-cohort-out validation.

Faithful re-implementation of the ICBClient / FedProx path in aws_run/pipeline.py without Flower
(non-private branch only: no Opacus, so plain Adam), with two preprocessing modes:

  identity_clip  : what the aws_results run actually did (no scaler file): raw log2 values,
                   NaN -> 0, clip to [-5, 5]. Used only to validate this re-implementation
                   against the reported S-NORM means (0.569 / 0.593 / 0.598).
  train_derived  : per-gene mean and SD estimated on the pooled training cohorts of each fold
                   (all samples of those cohorts, expression only, no labels) and applied
                   unchanged to the training clients, calibration partitions and the held-out
                   cohort. No clipping. Same scheme as crpr_sensitivity.py / matched_comparison.py.

Settings copied from pipeline.py CONFIG: epochs 5, rounds 15, lr 0.005, weight_decay 1e-4,
batch 8, proximal_mu 0.1 (squared L2), fit/cal split 80/20 stratified (random_state 42),
seeds 0-4, FedAvg-style weighting by client fit-partition size.
"""
import sys, numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

DATA = sys.argv[1]
CONFIG = {"epochs": 5, "rounds": 15, "lr": 0.005, "weight_decay": 1e-4, "batch_size": 8, "mu": 0.1}
TRAF = ["CD274", "CMTM6", "CMTM4", "DRG2", "RAB11A", "RAB8A", "SNX1", "SNX2", "SNX27", "VPS35"]
IFN = ["IFNG", "STAT1", "CXCL9", "CXCL10", "IDO1", "HLA-DRA"]
FS = {"Trafficking_Only": TRAF, "IFN_Gamma_Only": IFN, "Combined": TRAF + IFN}
COHORTS = ["Hugo", "Riaz", "Gide", "Liu", "IMvigor210"]
SEEDS = [0, 1, 2, 3, 4]

RAW = {c: pd.read_csv(f"{DATA}/{c}_clean.csv") for c in COHORTS}


def raw(c, f):
    d = RAW[c]
    return d[f].to_numpy(np.float32), d["response_binary"].to_numpy(np.float32)


def prep(X, mode, mean=None, std=None):
    if mode == "identity_clip":
        X = np.nan_to_num(X, nan=0.0, posinf=5.0, neginf=-5.0)
        return np.clip(X, -5.0, 5.0).astype(np.float32)
    return np.nan_to_num((X - mean) / std).astype(np.float32)


def split(X, y):
    Xf, Xc, yf, yc = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
    return Xf, yf


def fold(held, f, seed, mode):
    np.random.seed(seed); torch.manual_seed(seed)
    train = [c for c in COHORTS if c != held]
    mean = std = None
    if mode == "train_derived":
        Xp = np.vstack([raw(c, f)[0] for c in train])
        mean = np.nanmean(Xp, 0); std = np.nanstd(Xp, 0) + 1e-8
    clients = []
    for c in train:
        X, y = raw(c, f); Xf, yf = split(prep(X, mode, mean, std), y)
        dl = DataLoader(TensorDataset(torch.tensor(Xf), torch.tensor(yf).view(-1, 1)),
                        batch_size=CONFIG["batch_size"], shuffle=True, drop_last=False)
        clients.append((dl, len(Xf)))
    # Flower draws initial parameters from one freshly initialized client model.
    glob = [p.detach().clone() for p in nn.Linear(len(f), 1).parameters()]
    crit = nn.BCEWithLogitsLoss()
    for _ in range(CONFIG["rounds"]):
        upd, w = [], []
        for dl, n in clients:
            m = nn.Linear(len(f), 1)
            with torch.no_grad():
                for p, g in zip(m.parameters(), glob): p.copy_(g)
            opt = optim.Adam(m.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
            for _ in range(CONFIG["epochs"]):
                for xb, yb in dl:
                    opt.zero_grad()
                    loss = crit(m(xb), yb) + (CONFIG["mu"] / 2) * sum(
                        torch.sum((lp - g) ** 2) for lp, g in zip(m.parameters(), glob))
                    loss.backward(); opt.step()
            upd.append([p.detach().clone() for p in m.parameters()]); w.append(n)
        w = np.array(w, float) / sum(w)
        glob = [sum(wi * u[k] for wi, u in zip(w, upd)) for k in range(len(glob))]
    Xt, yt = raw(held, f); Xt = prep(Xt, mode, mean, std)
    m = nn.Linear(len(f), 1)
    with torch.no_grad():
        for p, g in zip(m.parameters(), glob): p.copy_(g)
        pt = torch.sigmoid(m(torch.tensor(Xt))).numpy().ravel()
    return roc_auc_score(yt, pt)


rows = []
for mode in sys.argv[2:]:
    for fn, f in FS.items():
        for h in COHORTS:
            for s in SEEDS:
                rows.append({"Mode": mode, "Feature_Set": fn, "Holdout": h, "Seed": s, "AUROC": fold(h, f, s, mode)})
df = pd.DataFrame(rows)
df.to_csv("snorm_rerun_results.csv", index=False)
cm = df.groupby(["Mode", "Feature_Set", "Holdout"]).AUROC.mean()
print(cm.unstack().round(3))
print("\nMean across cohorts (seed-averaged per cohort, then averaged):")
print(cm.groupby(["Mode", "Feature_Set"]).mean().round(3))
sd = df.groupby(["Mode", "Feature_Set", "Seed"]).AUROC.mean().groupby(["Mode", "Feature_Set"]).std()
print("\nSD of cohort-mean AUROC across seeds:"); print(sd.round(3))
