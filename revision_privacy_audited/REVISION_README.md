# Revision analyses (manuscript resubmission)

This folder contains code added in response to peer review. The original scripts in the repository root are retained unchanged, and **they produced all primary results reported in the manuscript** (main text, Tables 1 and 2, Figures 1 to 8, and Supplementary Tables S1 to S4).

## Where each reported result comes from

| Manuscript item | Script | Output |
|---|---|---|
| Main results, Tables S1 to S3 | `../pipeline.py`, `../pipeline_melanoma.py` (repository root) | `Comprehensive_Results*.csv`, `Coefficients_Detailed*.csv` |
| Table S4, Figures S3 and S4 | `../pipeline_recyc.py` (repository root) | as above, `_recyc` suffix |
| Table S5 (DRG2 ablation) | `drg2_incremental_value.py` | console output |
| Tables S6a and S6b (strict CR/PR endpoint) | `sensitivity/crpr_sensitivity.py` | `CR_PR_reclassification.csv`, `CR_PR_Sensitivity.csv` |
| Table S7 (tuning and optimization sensitivity) | `sensitivity/matched_comparison.py` (centralized columns) and `sensitivity/snorm_rerun.py` (federated column) | `Matched_Comparison.csv`, `snorm_rerun_results.csv` |
| Table S8 (training-derived standardization) | `sensitivity/snorm_rerun.py` | `snorm_rerun_results.csv` |
| Table S9 (per-client DP accounting) | `sensitivity/sdp_rebuild.py` | `Table_S-DP_published_run.csv` |

## Important: the `results/` folder

`results/` holds outputs from an exploratory run of the revised pipelines in this folder (`pipeline*.py`) **without** a public reference scaler. In that configuration, expression values were not standardized and were clipped to [-5, 5]. **These outputs are not the source of any result reported in the manuscript.** They are retained for transparency only.

Two consequences:

- The normalization sensitivity analysis (Table S8) was **not** taken from this run. It was rerun with training-derived standardization (`sensitivity/snorm_rerun.py`).
- `results/Privacy_Audit.csv` logs this exploratory run, not the published runs. Per-client accounting for the published runs is reconstructed in Table S9 (`sensitivity/sdp_rebuild.py`).

## Notes on the analyses

- **Table S8 and the federated column of Table S7.** `snorm_rerun.py` re-implements the non-private (ε = ∞) FedProx path of `../pipeline.py` without Flower, using the same model, Adam settings, 15 rounds × 5 local epochs, batch size 8, μ = 0.1, 80/20 fit/calibration split and seeds 0 to 4. Without differential privacy, Flower only performs weighted averaging of client weights, so this path is equivalent. As a check, the script also reproduces the exploratory unscaled-and-clipped setting to within 0.008 AUROC of the Flower run.
- **Table S9.** In the published runs, the noise multiplier was calibrated at a nominal sampling rate (batch size / fit-partition size), while Opacus executed Poisson sampling at 1 / (batches per epoch). `sdp_rebuild.py` recomputes realized ε from the executed rate and step count; realized ε is at or below each target for every client. Values are reconstructed from code settings, not from retained per-run logs.
- **Table S5** is a centralized analysis with one deterministic fit per fold, pooled held-out predictions and a 2000-replicate patient-level paired bootstrap (not stratified by cohort).
- **Table S7 centralized models** use all pooled training data; the federated model uses each cohort's 80% fitting partition.

## Scope of the privacy guarantee

Accounting covers DP-SGD training on standardized records within each training trajectory. Feature standardization, threshold selection, Platt recalibration, evaluation summaries and composition across multiple released runs (seeds, folds, feature sets) are outside the accounting. The federated runs are simulated. This is not an end-to-end differentially private workflow.

## Environment

Revision analyses (this folder): Python 3.11, numpy 1.26.4, pandas 2.2.2, scipy 1.13.1, scikit-learn 1.5.1, torch 2.2.2, opacus 1.5.2, flwr[simulation] 1.9.0, ray 2.10.0 (see `requirements.txt`).

The primary analyses (repository root) were run with the same code and the minimum versions in the root `requirements.txt`; the exact package versions of that run were not recorded.

```
conda create -y -n icb python=3.11 && conda activate icb
pip install -r requirements.txt
```

## Running the revision analyses

```
python drg2_incremental_value.py
ICB_DATA_DIR=./processed ICB_CLINICAL_DIR=./clinical python sensitivity/crpr_sensitivity.py
python sensitivity/matched_comparison.py
python sensitivity/snorm_rerun.py <folder with {Cohort}_clean.csv> train_derived identity_clip
python sensitivity/sdp_rebuild.py
```

## Data (not redistributed)

Patient-level expression and clinical data are not included. Each cohort is public; prepare `{Cohort}_clean.csv` (log2(x+1) expression of the 16 or 22 genes plus `response_binary`) from the original sources and place the files in `ICB_DATA_DIR`.

| Cohort | Expression | Response / RECIST source |
|---|---|---|
| Hugo | GEO GSE78220 (FPKM) | Hugo et al., Cell 2016, supplement mmc1 (sheet S1A, irRECIST) |
| Riaz | GEO GSE91061 (FPKM), pre-treatment samples | Riaz et al., Cell 2017, Table S2 |
| Gide | TIGER | TIGER clinical table |
| Liu | Liu et al., Nat Med 2019, Supplementary Data (TPM) | same paper, Supplementary Table 1 (BR) |
| IMvigor210 | IMvigor210CoreBiologies 1.0.1 | same package (Best Confirmed Overall Response) |

Patient identifiers are matched case-insensitively to the `{Cohort}_clean.csv` files.
