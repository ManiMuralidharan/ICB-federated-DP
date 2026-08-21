import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import flwr as fl
from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    f1_score, confusion_matrix, roc_curve, log_loss, matthews_corrcoef
)
from scipy.special import logit
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import numpy as np
import pandas as pd
import warnings
import hashlib
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings("ignore")

# =============================================================================
# 1. CONFIGURATION & SEEDING
# =============================================================================
PROTOTYPE_MODE = False  # set False for the full run once real CSVs are in place

CONFIG = {
    "epochs": 5, "rounds": 3 if PROTOTYPE_MODE else 15,
    "n_bootstraps": 100 if PROTOTYPE_MODE else 1000,
    "lr": 0.005, "weight_decay": 1e-4, "batch_size": 8,
    "dp_delta": 1e-5, "max_grad_norm": 1.2, "proximal_mu": 0.1,
}

TRAFFICKING_GENES = ["CD274", "CMTM6", "CMTM4", "DRG2", "RAB11A", "RAB8A",
                     "SNX1", "SNX2", "SNX27", "VPS35"]
IFN_GAMMA_GENES = ["IFNG", "STAT1", "CXCL9", "CXCL10", "IDO1", "HLA-DRA"]
ALL_FEATURES = TRAFFICKING_GENES + IFN_GAMMA_GENES
COHORTS = ["Hugo", "Riaz", "Gide", "Liu", "IMvigor210"]
FEATURE_SETS = {"Trafficking_Only": TRAFFICKING_GENES, "IFN_Gamma_Only": IFN_GAMMA_GENES, "Combined": ALL_FEATURES}

def set_seed(seed: int = 42):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

CAL_FRAC = 0.20   # held-out calibration partition of the development cohorts (for threshold + Platt)
STRICT_CAL_SPLIT = not PROTOTYPE_MODE   # real run: never silently reuse fit data as calibration data

def fit_cal_split(X, y, frac=CAL_FRAC, seed=42, cohort='?'):
    """Deterministic stratified split into a model-fit part and an OUT-OF-SAMPLE calibration part.
    The calibration part is excluded from model fitting and is used only to estimate the operating
    threshold and the Platt recalibration map -- never the external held-out cohort. Falls back to
    in-sample (documented) only when a cohort is too small/imbalanced to split."""
    from sklearn.model_selection import train_test_split as _tts
    try:
        Xf, Xc, yf, yc = _tts(X, y, test_size=frac, stratify=y, random_state=seed)
        if len(np.unique(yf)) < 2 or len(np.unique(yc)) < 2:
            raise ValueError("degenerate stratified calibration split")
        return Xf, yf, Xc, yc
    except ValueError as e:
        # GUARD: in the real analysis, a cohort too small/imbalanced to yield a held-out calibration
        # partition must force an EXPLICIT scientific decision -- never silently collapse to in-sample
        # calibration (which reintroduces the overfitting we removed).
        if STRICT_CAL_SPLIT:
            raise ValueError(f"[{cohort}] insufficient data/class balance for a held-out calibration "
                             f"split; decide explicitly (drop cohort, pool, or adjust CAL_FRAC).") from e
        warnings.warn(f"[prototype] calibration split fell back to in-sample for cohort={cohort}")
        return X, y, X, y

# =============================================================================
# 2. STATISTICS HELPERS  (DeLong test + Platt recalibration)  -- NEW
# =============================================================================
def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]: j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1; i = j
    T2 = np.empty(N); T2[J] = T
    return T2

def _fast_delong(preds_sorted_T, m):
    n = preds_sorted_T.shape[1] - m; k = preds_sorted_T.shape[0]
    pos = preds_sorted_T[:, :m]; neg = preds_sorted_T[:, m:]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r, :] = _midrank(pos[r, :]); ty[r, :] = _midrank(neg[r, :]); tz[r, :] = _midrank(preds_sorted_T[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n; v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    return aucs, cov

def delong_test(y_true, p1, p2):
    """Two-sided DeLong test for the difference between two correlated AUROCs."""
    y_true = np.asarray(y_true).astype(int)
    m = int(y_true.sum())
    if m == 0 or m == len(y_true): return np.nan, np.nan, np.nan
    order = np.argsort(-y_true)                       # positives first
    preds = np.vstack((np.asarray(p1), np.asarray(p2)))[:, order]
    aucs, cov = _fast_delong(preds, m)
    l = np.array([[1.0, -1.0]]); var = float(l @ cov @ l.T)
    if var <= 0: return aucs[0], aucs[1], np.nan
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return aucs[0], aucs[1], float(2 * stats.norm.sf(abs(z)))

def platt_recalibrate(p_train, y_train, p_test):
    """Platt scaling: fit logistic map on training predictions, apply to holdout."""
    e = 1e-6
    zt = logit(np.clip(np.asarray(p_train), e, 1 - e)).reshape(-1, 1)
    lr = LogisticRegression(penalty=None, solver='lbfgs')
    try:
        lr.fit(zt, np.asarray(y_train).astype(int))
        zte = logit(np.clip(np.asarray(p_test), e, 1 - e)).reshape(-1, 1)
        return lr.predict_proba(zte)[:, 1]
    except Exception:
        return np.asarray(p_test)

# =============================================================================
# 3. EVALUATION & METRICS ENGINE
# =============================================================================
def get_optimal_threshold(y_true, y_pred_prob):
    if len(np.unique(y_true)) < 2: return 0.5
    fpr, tpr, thr = roc_curve(y_true, y_pred_prob)
    return thr[np.argmax(tpr - fpr)]

def evaluate_predictions(y_true, y_pred_prob, threshold,
                         baseline_pred_prob=None, y_pred_recal=None):
    if len(np.unique(y_true)) < 2: return {}
    if baseline_pred_prob is not None:
        assert len(baseline_pred_prob) == len(y_true), "baseline/holdout misaligned"

    auroc = roc_auc_score(y_true, y_pred_prob)
    auprc = average_precision_score(y_true, y_pred_prob)
    brier = brier_score_loss(y_true, y_pred_prob)
    prevalence = float(np.mean(y_true))

    y_clip = np.clip(y_pred_prob, 1e-5, 1 - 1e-5)
    logloss = log_loss(y_true, y_clip)
    logits = logit(y_clip).reshape(-1, 1)
    calib_lr = LogisticRegression(penalty=None, solver='lbfgs')
    try:
        calib_lr.fit(logits, y_true)
        calib_slope = float(np.clip(calib_lr.coef_[0][0], -5.0, 5.0))   # winsorized (flag, not value)
        calib_intercept = float(calib_lr.intercept_[0])
    except Exception:
        calib_slope, calib_intercept = np.nan, np.nan

    y_bin = (y_pred_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_bin).ravel()
    sens = tp / (tp + fn + 1e-8); spec = tn / (tn + fp + 1e-8)
    f1 = f1_score(y_true, y_bin, zero_division=0); mcc = matthews_corrcoef(y_true, y_bin)

    # NEW: recalibrated calibration metrics (if recalibrated probs supplied)
    brier_recal = brier_score_loss(y_true, np.clip(y_pred_recal, 0, 1)) if y_pred_recal is not None else np.nan

    roc_s, prc_s, d_roc, d_prc = [], [], [], []
    for i in range(CONFIG["n_bootstraps"]):
        idx = resample(np.arange(len(y_true)), random_state=i)
        if len(np.unique(y_true[idx])) < 2: continue
        roc_s.append(roc_auc_score(y_true[idx], y_pred_prob[idx]))
        prc_s.append(average_precision_score(y_true[idx], y_pred_prob[idx]))
        if baseline_pred_prob is not None:
            d_roc.append(roc_s[-1] - roc_auc_score(y_true[idx], baseline_pred_prob[idx]))
            d_prc.append(prc_s[-1] - average_precision_score(y_true[idx], baseline_pred_prob[idx]))

    m = {
        "Prevalence": prevalence, "AUROC": auroc,
        "AUROC_CI_Low": np.percentile(roc_s, 2.5) if roc_s else np.nan,
        "AUROC_CI_High": np.percentile(roc_s, 97.5) if roc_s else np.nan,
        "AUPRC": auprc, "AUPRC_CI_Low": np.percentile(prc_s, 2.5) if prc_s else np.nan,
        "AUPRC_CI_High": np.percentile(prc_s, 97.5) if prc_s else np.nan,
        "Brier_Score": brier, "Brier_Recal": brier_recal, "Log_Loss": logloss,
        "Calibration_Slope": calib_slope, "Calibration_Intercept": calib_intercept,
        "Sensitivity": sens, "Specificity": spec, "F1": f1, "MCC": mcc,
    }
    if baseline_pred_prob is not None and d_roc:
        # NEW: DeLong p-value complements the bootstrap delta
        _, _, delong_p = delong_test(y_true, y_pred_prob, baseline_pred_prob)
        m.update({
            "Delta_AUROC": np.mean(d_roc), "Delta_AUROC_CI_Low": np.percentile(d_roc, 2.5),
            "Delta_AUROC_CI_High": np.percentile(d_roc, 97.5), "Delta_AUPRC": np.mean(d_prc),
            "Delta_AUPRC_CI_Low": np.percentile(d_prc, 2.5), "Delta_AUPRC_CI_High": np.percentile(d_prc, 97.5),
            "DeLong_p_vs_baseline": delong_p,
        })
    return m

# =============================================================================
# 4. DATA INGESTION  (robust to NaNs and absent genes)  -- FIXED
# =============================================================================
def load_and_normalize_cohort(cohort_name, features):
    import os as _os
    _data_dir = _os.environ.get("ICB_DATA_DIR", ".")
    _path = _os.path.join(_data_dir, f"{cohort_name}_clean.csv")
    if not _os.path.exists(_path):
        raise FileNotFoundError(
            f"Data file not found: {_path}\n"
            f"Set ICB_DATA_DIR to the folder with your *_clean.csv files, "
            f"e.g. export ICB_DATA_DIR=/path/to/processed")
    df = pd.read_csv(_path)
    missing = [g for g in features if g not in df.columns]
    if missing:
        raise ValueError(f"[{cohort_name}] missing gene columns: {missing}")
    X_raw = df[features].values.astype(np.float32)
    y_raw = df['response_binary'].values.astype(np.float32)
    mean = np.nanmean(X_raw, axis=0); std = np.nanstd(X_raw, axis=0) + 1e-8
    Xz = (X_raw - mean) / std
    Xz = np.nan_to_num(Xz, nan=0.0)
    return Xz, y_raw

# =============================================================================
# 5. FEDERATED LEARNING ARCHITECTURE
# =============================================================================
class ICBClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__(); self.net = nn.Linear(input_dim, 1)
    def forward(self, x): return self.net(x)

class ICBClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, noise_multiplier):
        self.model = model; self.train_loader = train_loader
        self.noise_multiplier = noise_multiplier; self.device = torch.device("cpu")
        self.criterion = nn.BCEWithLogitsLoss()
        self.original_state_keys = list(self.model.state_dict().keys())
    def get_parameters(self, config):
        sd = self.model.state_dict(); out = []
        for k in self.original_state_keys:
            key = f"_module.{k}" if f"_module.{k}" in sd else k; out.append(sd[key].cpu().numpy())
        return out
    def set_parameters(self, parameters):
        sd = self.model.state_dict()
        for k, p in zip(self.original_state_keys, parameters):
            key = f"_module.{k}" if f"_module.{k}" in sd else k; sd[key] = torch.tensor(p, device=self.device)
        self.model.load_state_dict(sd, strict=True)
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        opt = optim.Adam(self.model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
        if self.noise_multiplier > 0:
            # DP GUARD: make_private with a PRE-COMPUTED noise_multiplier (derived once for the whole
            # trajectory in run_federated_fold), NOT make_private_with_epsilon. The latter re-derives
            # noise from epsilon for THIS round's epochs only, silently under-counting the cumulative
            # budget ~R-fold over R rounds. Do NOT "simplify" this to _with_epsilon.
            pe = PrivacyEngine()
            self.model, opt, loader = pe.make_private(module=self.model, optimizer=opt,
                data_loader=self.train_loader, noise_multiplier=self.noise_multiplier, max_grad_norm=CONFIG["max_grad_norm"])
        else:
            loader = self.train_loader
        mu = config.get("proximal_mu", 0.0)
        gw = [torch.tensor(p, device=self.device) for p in parameters]  # global weights BEFORE Opacus wraps model
        self.model.train()
        epochs_run = 0
        for _ in range(CONFIG["epochs"]):
            epochs_run += 1
            for Xb, yb in loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device); opt.zero_grad()
                loss = self.criterion(self.model(Xb), yb)
                if mu > 0:
                    loss += (mu / 2) * sum(torch.norm(lw - g) ** 2 for lw, g in zip(self.model.parameters(), gw))
                loss.backward(); opt.step()
        # DP GUARD: local passes actually run MUST equal CONFIG["epochs"]; the accountant assumes
        # epochs = CONFIG["epochs"] * CONFIG["rounds"]. If this fires, the reported epsilon is wrong.
        assert epochs_run == CONFIG["epochs"], "local epochs run != CONFIG['epochs'] used in DP accounting"
        # 2nd return value = aggregation weight (client sample count) -> aggregation is SIZE-weighted.
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

class SaveModelStrategy(fl.server.strategy.FedProx):
    def __init__(self, *a, **k): super().__init__(*a, **k); self.final_weights = None
    def aggregate_fit(self, rnd, results, failures):
        agg, met = super().aggregate_fit(rnd, results, failures)
        if agg is not None: self.final_weights = fl.common.parameters_to_ndarrays(agg)
        return agg, met

# =============================================================================
# 6. EXPERIMENT RUNNERS  (now return recalibrated probabilities)
# =============================================================================
def run_federated_fold(held_out, features, target_epsilon, strategy_type, seed=42):
    set_seed(seed); train_cohorts = [c for c in COHORTS if c != held_out]
    # DP GUARD: single source of truth for trajectory length. The accountant (below) and the server's
    # num_rounds MUST use these same values or the reported epsilon desyncs from what is actually run.
    num_rounds = CONFIG["rounds"]
    dp_epochs = CONFIG["epochs"] * num_rounds        # total local passes each client makes over the run
    def client_fn(cid):
        cohort = train_cohorts[int(cid)]; X, y = load_and_normalize_cohort(cohort, features)
        Xf, yf, _, _ = fit_cal_split(X, y, cohort=cohort)          # client trains on the model-fit part only
        # drop_last=False: cleaner sampling-rate interpretation for the non-private path; under DP,
        # Opacus replaces this loader with Poisson subsampling anyway (drop_last is ignored there).
        tl = DataLoader(TensorDataset(torch.tensor(Xf), torch.tensor(yf).view(-1, 1)),
                        batch_size=CONFIG["batch_size"], shuffle=True, drop_last=False)
        if target_epsilon < float('inf'):
            sr = CONFIG["batch_size"] / len(Xf); td = min(CONFIG["dp_delta"], 1.0 / len(Xf))
            alphas = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 128))
            # sigma derived ONCE for the full dp_epochs trajectory, then held fixed every round (see fit()).
            nm = get_noise_multiplier(target_epsilon=target_epsilon, target_delta=td, sample_rate=sr,
                                      epochs=dp_epochs, accountant="rdp", alphas=alphas)
        else:
            nm = 0.0
        return ICBClient(ICBClassifier(len(features)), tl, nm)
    strategy = SaveModelStrategy(
        # DP GUARD: fraction_fit MUST stay 1.0 -- every client trains every round, which the
        # single-trajectory accounting assumes. Client subsampling would need a participation term.
        proximal_mu=CONFIG["proximal_mu"] if strategy_type == "FedProx" else 0.0,
        fraction_fit=1.0, fraction_evaluate=0.0, min_available_clients=len(train_cohorts),
        on_fit_config_fn=lambda rnd: {"proximal_mu": CONFIG["proximal_mu"] if strategy_type == "FedProx" else 0.0})
    fl.simulation.start_simulation(client_fn=client_fn, num_clients=len(train_cohorts),
        config=fl.server.ServerConfig(num_rounds=num_rounds), strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0})

    X_test, y_test = load_and_normalize_cohort(held_out, features)
    model = ICBClassifier(len(features))
    if strategy.final_weights:
        model.load_state_dict({k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), strategy.final_weights)})
    model.eval()
    # Pooled OUT-OF-SAMPLE calibration partition of the training cohorts (excluded from federated fitting).
    Xcal = np.vstack([fit_cal_split(*load_and_normalize_cohort(c, features), cohort=c)[2] for c in train_cohorts])
    ycal = np.concatenate([fit_cal_split(*load_and_normalize_cohort(c, features), cohort=c)[3] for c in train_cohorts])
    with torch.no_grad():
        p_cal  = torch.sigmoid(model(torch.tensor(Xcal))).numpy().ravel()
        p_test = torch.sigmoid(model(torch.tensor(X_test))).numpy().ravel()
    # LEAK GUARD: the operating threshold and the Platt map are fit on the held-out CALIBRATION partition
    # (ycal, p_cal) -- out-of-sample for the model, and never the external test cohort. Do not substitute
    # in-sample training predictions (overfits the calibration map) or y_test/p_test (leaks eval labels).
    thr = get_optimal_threshold(ycal, p_cal)
    p_test_recal = platt_recalibrate(p_cal, ycal, p_test)
    coeffs = model.net.weight.detach().cpu().numpy().flatten()
    return y_test, p_test, thr, coeffs, p_test_recal

def run_centralized_baselines(held_out, features):
    set_seed()
    X_all = np.vstack([load_and_normalize_cohort(c, features)[0] for c in COHORTS if c != held_out])
    y_all = np.concatenate([load_and_normalize_cohort(c, features)[1] for c in COHORTS if c != held_out])
    X_test, y_test = load_and_normalize_cohort(held_out, features)

    # Fit on the model-fit partition; calibrate on the held-out calibration partition (parity with FL).
    Xf, yf, Xcal, ycal = fit_cal_split(X_all, y_all, cohort=held_out+'-dev')
    m_global = LogisticRegressionCV(Cs=10, cv=5, penalty='l2', max_iter=1000).fit(Xf, yf)
    p_global = m_global.predict_proba(X_test)[:, 1]
    p_cal = m_global.predict_proba(Xcal)[:, 1]
    # LEAK GUARD: threshold + Platt fit on the OUT-OF-SAMPLE calibration partition (ycal, p_cal), applied
    # to the held-out cohort (never y_test, never the model-fit data used to train m_global).
    global_thresh = get_optimal_threshold(ycal, p_cal)
    p_global_recal = platt_recalibrate(p_cal, ycal, p_global)

    # local control -- split now guarded so a tiny/imbalanced cohort can't crash the sweep  -- FIXED
    try:
        Xtr_l, Xte_l, ytr_l, yte_l = train_test_split(X_test, y_test, test_size=0.2, stratify=y_test, random_state=42)
        m_local = LogisticRegression(max_iter=1000).fit(Xtr_l, ytr_l)
        loc_thresh = get_optimal_threshold(ytr_l, m_local.predict_proba(Xtr_l)[:, 1])
        p_local = m_local.predict_proba(Xte_l)[:, 1]
    except ValueError:
        yte_l, p_local, loc_thresh = y_test, np.full_like(y_test, float(np.mean(y_test))), 0.5

    return (y_test, p_global, global_thresh, m_global, p_global_recal), (yte_l, p_local, loc_thresh)

# =============================================================================
# 7. MAIN SWEEP
# =============================================================================
if __name__ == "__main__":
    epsilons = [float('inf'), 10.0, 8.0, 4.0, 2.0, 1.0] if not PROTOTYPE_MODE else [float('inf'), 4.0]
    strategies = ["FedAvg", "FedProx"] if not PROTOTYPE_MODE else ["FedProx"]
    SEEDS = [0, 1, 2, 3, 4] if not PROTOTYPE_MODE else [42]   # repeat FL runs over seeds (DP/init/minibatch stochasticity)
    all_results, all_predictions, all_coefficients = [], [], []
    # ---- progress tracking ----
    import time as _time
    _total = len(FEATURE_SETS) * len(COHORTS) * len(strategies) * len(epsilons) * len(SEEDS)
    _done = 0
    _t0 = _time.time()
    def _tick(feat_name, holdout, strat, eps, seed):
        global _done
        _done += 1
        pct = 100.0 * _done / _total
        el = (_time.time() - _t0) / 60.0
        eta = (el / _done) * (_total - _done) if _done else 0.0
        es = 'inf' if eps == float('inf') else f'{eps:g}'
        print(f"[{_done:4d}/{_total} | {pct:5.1f}%] {feat_name:16s} holdout={holdout:10s} "
              f"{strat:7s} eps={es:>3s} seed={seed} | elapsed={el:5.1f}m ETA={eta:5.1f}m", flush=True)

    print(f"Starting sweep (prototype={PROTOTYPE_MODE}, seeds={SEEDS})")

    # --- NEW: pooled standardized expression for the PCA heterogeneity figure ---
    pooled = []
    for c in COHORTS:
        Xc, yc = load_and_normalize_cohort(c, ALL_FEATURES)
        for i in range(len(yc)):
            pooled.append({"Cohort": c, "response_binary": int(yc[i]),
                           **{g: Xc[i, j] for j, g in enumerate(ALL_FEATURES)}})
    pd.DataFrame(pooled).to_csv("Expression_Pooled.csv", index=False)

    for feat_name, feat_list in FEATURE_SETS.items():
        print(f"\n=== {feat_name} ===")
        for holdout in COHORTS:
            print(f"  holdout={holdout}")
            (y_c, p_c, t_c, m_c, p_c_recal), (y_l, p_l, t_l) = run_centralized_baselines(holdout, feat_list)

            for yt, yp, yr in zip(y_c, p_c, p_c_recal):
                all_predictions.append({"Feature_Set": feat_name, "Holdout": holdout, "Model": "Centralized_Reference",
                                        "Cumulative_Epsilon": np.inf, "Seed": "deterministic", "y_true": yt, "y_pred_prob": yp, "y_pred_recal": yr})
            for g, w in zip(feat_list, m_c.coef_[0]):
                all_coefficients.append({"Feature_Set": feat_name, "Holdout": holdout, "Model": "Centralized_Reference",
                                         "Cumulative_Epsilon": np.inf, "Seed": "deterministic", "Gene": g, "Weight": w})
            all_results.append({"Feature_Set": feat_name, "Holdout": holdout, "Model": "Centralized_Reference",
                                "Seed": "deterministic", **evaluate_predictions(y_c, p_c, t_c, y_pred_recal=p_c_recal)})

            # Within-cohort local baseline: trained on 80% of the TARGET cohort -> NOT an external comparator.
            loc_m = evaluate_predictions(y_l, p_l, t_l)
            if loc_m:
                all_results.append({"Feature_Set": feat_name, "Holdout": holdout, "Model": "Within_Cohort_Local_Baseline",
                                    "Seed": "deterministic", **loc_m})
                for yt, yp in zip(y_l, p_l):
                    all_predictions.append({"Feature_Set": feat_name, "Holdout": holdout, "Model": "Within_Cohort_Local_Baseline",
                                            "Cumulative_Epsilon": np.inf, "Seed": "deterministic", "y_true": yt, "y_pred_prob": yp, "y_pred_recal": yp})

            for strat in strategies:
                for eps in epsilons:
                    for seed in SEEDS:
                        y_f, p_f, t_f, coef_f, p_f_recal = run_federated_fold(holdout, feat_list, eps, strat, seed=seed)
                        _tick(feat_name, holdout, strat, eps, seed)
                        # store per-sample predictions only for the first seed (keeps the file manageable)
                        if seed == SEEDS[0]:
                            for yt, yp, yr in zip(y_f, p_f, p_f_recal):
                                all_predictions.append({"Feature_Set": feat_name, "Holdout": holdout, "Model": f"FL_{strat}",
                                                        "Cumulative_Epsilon": eps, "Seed": seed, "y_true": yt, "y_pred_prob": yp, "y_pred_recal": yr})
                        for g, w in zip(feat_list, coef_f):
                            all_coefficients.append({"Feature_Set": feat_name, "Holdout": holdout, "Model": f"FL_{strat}",
                                                     "Cumulative_Epsilon": eps, "Seed": seed, "Gene": g, "Weight": w})
                        all_results.append({"Feature_Set": feat_name, "Holdout": holdout, "Model": f"FL_{strat}",
                                            "Cumulative_Epsilon": eps, "Seed": seed,
                                            **evaluate_predictions(y_f, p_f, t_f, baseline_pred_prob=p_c, y_pred_recal=p_f_recal)})

    pd.DataFrame(all_results).to_csv("Comprehensive_Results.csv", index=False)
    pd.DataFrame(all_predictions).to_csv("Predictions_Detailed.csv", index=False)
    pd.DataFrame(all_coefficients).to_csv("Coefficients_Detailed.csv", index=False)
    print(f"\n[100.0%] DONE. {_done}/{_total} folds in {(_time.time()-_t0)/60.0:.1f} min.", flush=True)
    print("\nSaved Comprehensive_Results.csv, Predictions_Detailed.csv, Coefficients_Detailed.csv, Expression_Pooled.csv")
