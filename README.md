# Cancer type classification research demo

This project is an educational research prototype that uses gene-expression profiles to classify samples among five cancer classes. It includes a reproducible training/evaluation script and a Streamlit app for exploring held-out examples or scoring a CSV containing the selected gene features.

> **Research use only:** This project is not clinically validated. Do not use it to diagnose, screen, or guide treatment.

## What the app does

- Predicts one of five dataset classes: BRCA, COAD, KIRC, LUAD, or PRAD.
- Selects 20 gene-expression features using training data only.
- Compares Random Forest and Extra Trees using validation macro F1.
- Reports accuracy, balanced accuracy, macro F1, per-class results, and a confusion matrix on a separate held-out test set.
- Provides model-level feature rankings and optional local SHAP explanations.
- Accepts CSV uploads with the 20 selected `gene_<number>` columns.

Feature IDs are dataset column names, not official gene symbols. The repository does not currently include an annotation mapping them to gene symbols. Feature importance and SHAP describe model behavior; they do not prove causation or clinical relevance.

## Data and evaluation

The source is the TCGA Pan-Cancer RNA-Seq dataset identified in the notebook as `TCGA-PANCAN-HiSeq-801x20531`, containing 801 samples and 20,531 expression features across five classes.

The deployment artifacts were built with a stratified 60/20/20 train/validation/test workflow. Feature selection is fit on the training subset; model selection uses validation macro F1; the held-out test split is reserved for final evaluation. The current report is in `deployment_metrics.json`. Rebuild the artifacts to reproduce the split and report.

The reported test metrics measure performance on one split from this dataset, not performance on an independent cohort, hospital, or clinical population.

## Run locally

Python 3.14 and the pinned dependencies in `requirements.txt` are used for the current model artifact.

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

On Windows PowerShell:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

## Rebuild deployment artifacts

The small files `deployment_model.joblib`, `deployment_demo.csv`, `deployment_features.csv`, and `deployment_metrics.json` are the app's runtime assets. The raw source dataset is only needed to rebuild them.

If the source archive is available, extract it and run the builder:

```bash
tar -xzf data/cancer_project/TCGA-PANCAN-HiSeq-801x20531.tar.gz \
  -C data/cancer_project
python build_deployment_artifacts.py
```

The builder checks that expression and label files have matching, unique sample IDs before training. It writes compact assets suitable for the app, avoiding the need to deploy the 150+ MB expression matrix.

## Deploy on Streamlit Community Cloud

The GitHub repository is public, so an app deployed from it will also be publicly accessible.

**Live app:** [cancer-biomarker-ai-pipeline.streamlit.app](https://cancer-biomarker-ai-pipeline.streamlit.app/)

1. Push the app code, pinned `requirements.txt`, and all four deployment assets to GitHub.
2. Sign in at [Streamlit Community Cloud](https://share.streamlit.io/) with GitHub.
3. Choose **Create app**, then select `marowa-labs/cancer-biomarker-ai-pipeline`, branch `main`, and entrypoint `app.py`.
4. In Advanced settings, choose Python 3.14 to match the artifact build environment.
5. Deploy and wait for the build logs to report success.

No secrets are required by the app. Do not commit credentials or patient-identifiable data.

## Main project files

| File | Purpose |
|---|---|
| `app.py` | Streamlit research demo |
| `build_deployment_artifacts.py` | Validates source alignment, selects features, compares models, evaluates a holdout, and writes app assets |
| `deployment_model.joblib` | Selected fitted model and feature metadata |
| `deployment_demo.csv` | Compact held-out demo samples |
| `deployment_features.csv` | Training-only feature ranking |
| `deployment_metrics.json` | Validation comparison and held-out test report |
| `cancer_classification.ipynb` | Exploratory training and analysis notebook |
| `VIVA_PREPARATION.md` | Viva questions, answers, and limitations |

## Further work

- Add a verified annotation file mapping dataset feature IDs to official gene symbols.
- Validate on an independent cohort and assess dataset/batch effects.
- Compare additional models with nested or repeated validation.
- Improve input checks for expression normalization and dataset provenance.
- Treat all feature rankings as research hypotheses until experimentally validated.
