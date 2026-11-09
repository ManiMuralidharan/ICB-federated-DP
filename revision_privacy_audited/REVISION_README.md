# Revised, privacy-audited pipeline and sensitivity analyses

This folder contains the revised version of the analysis, produced in response to
peer-review feedback on the differential-privacy accounting and the validation
design, and after discussion with a privacy expert. The original scripts in the
repository root are retained unchanged for provenance; nothing here replaces them.

The revised code makes the differential-privacy accounting auditable and adds the
sensitivity analyses requested by the reviewers. It was run on a clean cloud
instance; the summary outputs under `results/` are the files the manuscript and
supplementary tables are built from.

## What changed from the original pipeline

- **Per-client DP audit.** Each client's sampling rate, noise multiplier, executed
  DP-SGD optimizer steps and realized epsilon are logged every round and composed
  with a Renyi-DP accountant, written to `results/Privacy_Audit.csv`. The realized
  epsilon sits at or just below each target budget; smaller cohorts take a larger
  noise multiplier and fewer steps to reach the same budget.
- **Noise calibration** uses the exact Opacus loader sampling rate `1/len(loader)`
  and the planned optimizer-step count; **Poisson sampling** is explicit.
- **Standardization** no longer uses private cohort statistics by default (fixed
  record-wise preprocessing; an optional public reference scaler via
  `DP_PUBLIC_SCALER_NPZ`). This is also the basis of the normalization sensitivity
  analysis.
- **FedProx** uses the squared-L2 proximal term.
- **Sensitive patient-level exports are off by default** and require
  `EXPORT_SENSITIVE_PATIENT_DATA=YES`; these files are never published under a DP
  claim.
- No silent synthetic-data fallback: a missing cohort or gene raises an error.

## Scope of the privacy guarantee

The reported (epsilon, delta = 1e-5) applies to the federated model-training
updates. Feature standardization, operating-threshold selection and recalibration
are data-dependent and are **outside** the privacy accounting. The federated runs
are conducted in simulation. This is **not** an end-to-end differentially private
workflow, and we do not describe it as one.

## Files

```
pipeline.py                     primary analysis (5 cohorts, 16 genes)
pipeline_melanoma.py            melanoma-only sensitivity (4 cohorts)
pipeline_recyc.py               extended panel (adds recycling genes)
pipeline_melanoma_recyc.py      extended panel, melanoma-only
pipeline_melanoma_recyc_tracked.py   as above, with progress logging
dp_reconstruct.py               independent RDP reconstruction of realized epsilon
drg2_incremental_value.py       IFN-gamma vs IFN-gamma + DRG2 ablation (Table S5)
sensitivity/crpr_sensitivity.py CR/PR-only endpoint sensitivity (Tables S6a, S6b)
requirements.txt                pinned dependencies
setup_and_run.sh                one-shot environment setup + run (Ubuntu)
results/                        Privacy_Audit.csv, Comprehensive_Results.csv,
                                Coefficients_Detailed.csv  (no patient-level data)
```

## Environment (important)

Use **Python 3.11**. Python 3.13 and 3.14 do not yet have wheels for the pinned
scientific stack, and building from source fails. The reliable setup is Miniforge
plus the pinned `requirements.txt`:

```bash
conda create -y -n icb python=3.11 && conda activate icb
pip install -r requirements.txt
```

Pinned versions: numpy 1.26.4, pandas 2.2.2, scipy 1.13.1, scikit-learn 1.5.1,
torch 2.2.2, opacus 1.5.2, flwr[simulation] 1.9.0, ray 2.11.0.

## Running

```bash
# main sweep (writes results/*.csv and Privacy_Audit.csv)
python pipeline.py

# verify the privacy accounting independently
python dp_reconstruct.py

# DRG2 incremental-value ablation (Table S5)
python drg2_incremental_value.py

# CR/PR endpoint sensitivity (Tables S6a, S6b) -- needs the clinical sources below
ICB_DATA_DIR=./processed ICB_CLINICAL_DIR=./clinical python sensitivity/crpr_sensitivity.py
```

## Data: not redistributed here

Patient-level expression and clinical data are **not** included in this repository.
Each cohort is public; obtain and prepare it from its original publication, then
place the processed `{Cohort}_clean.csv` files in the folder given by `ICB_DATA_DIR`.

The CR/PR sensitivity analysis additionally needs the original RECIST categories,
which come from each cohort's published supplement (place them in
`ICB_CLINICAL_DIR`):

| Cohort | RECIST source | Column |
|---|---|---|
| Hugo | Hugo et al., Cell 2016, supplement mmc1 (sheet S1A) | irRECIST |
| Riaz | Riaz et al., Cell 2017, Table S2 | Response |
| Gide | Gide et al., Cancer Cell 2019, clinical table | response |
| Liu | Liu et al., Nature Medicine 2019, Supplementary Table 1 | BR |
| IMvigor210 | Mariathasan et al., Nature 2018 (IMvigor210CoreBiologies) | Best Confirmed Overall Response |

Patient identifiers are matched case-insensitively to the `{Cohort}_clean.csv`
files.
