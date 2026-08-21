# Federated and Differentially Private Prediction of Immune Checkpoint Blockade Response

This repository contains the analysis code for a study that asks two questions:

1. **Can we predict which cancer patients respond to immunotherapy without ever pooling their private data in one place?**
2. **Do genes involved in "PD-L1 trafficking" help that prediction, beyond the well-known inflammation genes?**

The short answers: **(1) yes** — privacy-preserving training works and barely costs any accuracy, and **(2) no** — the trafficking genes don't improve prediction overall, but one of them, **DRG2**, behaves in a consistent, biologically sensible way worth following up.

> This code accompanies a manuscript submitted to *AI in Medicine* (MDPI). It is shared for transparency and reproducibility.

---

## What this project does, in plain language

Imagine five hospitals, each with gene-activity data from cancer patients treated with immunotherapy. Privacy rules mean the hospitals can't share raw patient data. So instead of copying everyone's data into one place, this project trains a prediction model in a **federated** way: each hospital trains on its own data locally and only shares the *math it learned* (never the patients). A coordinator blends those lessons into one shared model.

On top of that, it adds **differential privacy** — a controlled amount of mathematical "noise" so that no individual patient can be reverse-engineered from the model. The privacy strength can be dialed up or down (the "epsilon" values you'll see).

Then it tests the model fairly: train on four hospitals, predict the fifth (one the model has never seen). Because the model is simple and interpretable, you can look inside it and see which genes it relies on — which is how the DRG2 finding emerged.

---

## Repository contents

```
.
├── README.md              <- you are here
├── LICENSE
├── requirements.txt       <- Python packages needed
├── .gitignore             <- keeps patient data OUT of git
├── src/
│   ├── pipeline.py            <- main analysis: 5 cohorts, 16 genes
│   ├── pipeline_melanoma.py   <- sensitivity analysis: 4 melanoma cohorts only
│   ├── pipeline_recyc.py      <- extended analysis: 5 cohorts, 22 genes (adds recycling genes)
│   └── make_figures.R         <- generates the figures from the results
└── figures/               <- (figures land here when you run make_figures.R)
```

The three `pipeline_*.py` scripts are near-identical; they differ only in which cohorts and genes they use:

| Script | Cohorts | Genes | Purpose |
|---|---|---|---|
| `pipeline.py` | all 5 | 16 | primary analysis |
| `pipeline_melanoma.py` | 4 (melanoma only) | 16 | check the result holds without the urothelial cohort |
| `pipeline_recyc.py` | all 5 | 22 | check DRG2 holds when 6 recycling genes are added |

---

## About the data

**Patient data is NOT included in this repository.** Each of the five cohorts (Hugo, Riaz, Gide, Liu, IMvigor210) is publicly available from its original publication. You obtain and prepare them yourself, then place the prepared files where the scripts can find them.

Each prepared file is a plain CSV called `{Cohort}_clean.csv` (or `{Cohort}_clean_recyc.csv` for the 22-gene version), with one row per patient: a patient ID, one column per gene, and a final `response_binary` column (1 = responded, 0 = did not).

---

## How to run it (step by step, no prior experience assumed)

### Step 1 — Install Python and the required packages

You need **Python 3.10, 3.11, or 3.12** (not 3.13+ yet — some packages aren't ready for it). Check your version:

```bash
python --version
```

Then set up a clean workspace and install everything:

```bash
# create an isolated environment so this doesn't affect your other Python projects
python -m venv venv

# turn it on:
#   Mac/Linux:
source venv/bin/activate
#   Windows:
venv\Scripts\activate

# install the packages
pip install -r requirements.txt
```

### Step 2 — Put your prepared data files in a folder

Place your `*_clean.csv` files (and `*_clean_recyc.csv` for the extended run) in one folder, for example a folder called `processed`.

### Step 3 — Tell the scripts where that folder is

The scripts look for an environment variable called `ICB_DATA_DIR`. Point it at your data folder:

```bash
#   Mac/Linux:
export ICB_DATA_DIR="/full/path/to/processed"
#   Windows (Command Prompt):
set ICB_DATA_DIR=C:\full\path\to\processed
```

If you skip this step, the script will stop with a clear message telling you to set it — it will **not** guess or make up data.

### Step 4 — Run an analysis

```bash
# main analysis (5 cohorts, 16 genes)
python src/pipeline.py

# melanoma-only sensitivity analysis
python src/pipeline_melanoma.py

# extended 22-gene analysis
python src/pipeline_recyc.py
```

While it runs, you'll see a **live progress bar** like this:

```
[  42/720 |   5.8%] Combined         holdout=Gide       FedProx eps=  4 seed=2 | elapsed= 18.2m ETA=283.5m
```

That tells you: how many steps are done, the percentage, which part it's working on, how long it's been running, and roughly how long is left. When it's finished you'll see `[100.0%] DONE`.

**This takes a while** (a few hours) because it trains hundreds of small models. That's normal.

### Step 5 — Keep long runs alive (optional but recommended)

Because a run takes hours, you don't want it to die if your computer sleeps or you close the terminal.

```bash
# Mac:  keeps the Mac awake and keeps running even if the terminal closes
caffeinate -i nohup python src/pipeline.py > log.txt 2>&1 &
#   then watch progress with:
tail -f log.txt

# Linux: same idea without caffeinate
nohup python src/pipeline.py > log.txt 2>&1 &

# Windows: stop the PC sleeping first, then run in a window you leave open
powercfg /change standby-timeout-ac 0
python src\pipeline.py
```

### Step 6 — Make the figures

Once an analysis finishes, it writes result files (`Comprehensive_Results.csv`, `Coefficients_Detailed.csv`, etc.) into your data folder. Then generate the figures:

```bash
Rscript src/make_figures.R
```

(You need R installed, plus the packages the script loads at the top — install those in R with `install.packages(c("ggplot2", ...))`.)

---

## What comes out

Each run produces four CSV files:

| File | What's in it |
|---|---|
| `Comprehensive_Results*.csv` | accuracy numbers (AUROC, etc.) for every condition |
| `Coefficients_Detailed*.csv` | how much weight the model gave each gene (this is where the DRG2 finding lives) |
| `Predictions_Detailed*.csv` | the model's prediction for each patient |
| `Expression_Pooled*.csv` | the standardized gene data, used for the overview figure |

The `*_melanoma` and `*_recyc` versions come from the melanoma-only and 22-gene scripts respectively.

---

## A note on honesty about the results

This project deliberately uses a **simple, interpretable model** and reports **modest accuracy** (around 0.56–0.61 AUROC, where 0.50 is a coin flip). It is a **feasibility and interpretability study**, not a ready-to-use clinical tool. The DRG2 finding is **hypothesis-generating** — a lead worth investigating further, not a proven biomarker. Please read the accompanying manuscript for the full, careful interpretation.

---

## License

Released under the MIT License. See [LICENSE](LICENSE).
