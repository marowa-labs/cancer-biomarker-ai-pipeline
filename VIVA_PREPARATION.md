# Cancer classification project: viva preparation

This guide summarizes what the project currently does and provides concise answers to likely viva questions. Keep the limitations clear: this is a research and educational prototype, not a clinically validated diagnostic system.

## A short project summary

> This project is an educational research prototype. It uses 20 selected gene-expression features to classify samples among five cancer types. We compare Random Forest and Extra Trees on a validation split, then report results on a separate held-out test split. The test results are promising on this dataset, but gene identifiers still need biological annotation and findings require independent biological and clinical validation.

## Likely viva questions

### What does the model do?

The model takes a sample's gene-expression values and predicts which of five cancer classes it most resembles. The project also ranks gene features that influenced the model. It predicts a cancer class; it does not prove that a gene causes cancer or establish that a gene is a clinically validated biomarker.

### Where did the model come from?

It is not a pretrained model downloaded from a model hub. The reproducible builder selects features from the training portion of the TCGA Pan-Cancer RNA-Seq data, compares tree-based models on a validation split, and saves the selected fitted model and metadata as `deployment_model.joblib`. The Streamlit app loads this compact artifact. The build process is in `build_deployment_artifacts.py`.

The exploratory notebook is `cancer_classification.ipynb`; the current app uses `deployment_model.joblib`.

### What does `gene_<number>` mean? Which gene is it?

`gene_<number>` is a feature identifier in this dataset, not an official gene symbol. The project files currently do not provide an annotation mapping from these feature identifiers to biological gene names. We should not guess their identities. To name a feature, we need the correct annotation table for this exact dataset and must confirm the feature ordering.

The 20 features selected for the deployment model are listed in `deployment_features.csv`.

### What does the sample index mean?

The app now selects by the dataset's sample ID rather than an unexplained numerical index. The demonstration samples come from the held-out test set; their observed labels are available so a viewer can compare the prediction with the known dataset class.

The builder checks that the full source expression and label files have matching, unique sample IDs, then generates a compact 161-sample held-out demo file. The legacy `X_cleaned.csv` and `y.csv` files are not used by the current app.

### Which cancers can it predict?

The saved model has five output classes:

| Code | Cancer type |
|---|---|
| BRCA | Breast invasive carcinoma |
| COAD | Colon adenocarcinoma |
| KIRC | Kidney renal clear cell carcinoma |
| LUAD | Lung adenocarcinoma |
| PRAD | Prostate adenocarcinoma |

The model can only select among these five classes. It does not cover every cancer type. Use the label definitions associated with the exact dataset when giving formal names.

### Why use one model instead of comparing several?

Random Forest and Extra Trees are compared using the same selected features and validation samples, with macro F1 as the selection metric. The selected model is recorded in `deployment_metrics.json`. The held-out test set is not used to select the model. This is a more informative comparison than reporting just one algorithm, though a larger study should include repeated or nested validation and other appropriate baselines.

### How do you know the model works accurately?

The current deployment builder reports performance on a stratified held-out test set of 161 samples. The latest generated report records approximately **98.1% accuracy**, **97.9% balanced accuracy**, and **98.4% macro F1**. It also includes per-class precision, recall, F1, and a confusion matrix in `deployment_metrics.json`. These are internal results from one split of this dataset, not proof of performance on new hospitals, populations, or clinical samples. The notebook's older scores use a different model and evaluation workflow; quote the deployment report when discussing the current app.

### What do the bars in the dashboard mean?

The app separates two charts:

- **Model-level feature ranking** shows the training-only Extra Trees importance scores used to select the 20 features. It is not a sample-specific explanation.
- **SHAP explanation** is generated for a selected sample only when the viewer clicks “Explain this prediction with SHAP.” Signed values show how features move the model output for the selected class.

Neither chart proves causation or establishes clinical biomarker utility. Explain which chart you are showing.

### How could the takeaways help researchers?

The ranked genes can help researchers prioritize candidates for follow-up, such as checking the literature, testing an independent dataset, or designing laboratory experiments. They are hypotheses for further research, not treatment recommendations or validated diagnostic findings.

### Was the app deployed?

The app is configured for Streamlit Community Cloud, but it is not publicly deployed until the deployment assets and code are pushed to GitHub and an app is created at Streamlit Community Cloud. A live app from this public repository will be publicly accessible.

### Which tools and modules did you use?

- **Python** for the pipeline and app.
- **pandas** and **NumPy** for data handling.
- **scikit-learn** for Extra Trees/Random Forest training and evaluation.
- **SHAP** for model explanation.
- **joblib** for saving and loading the model.
- **Streamlit** for the interactive dashboard.

The dependency list is in `requirements.txt`.

## Honest limitations to acknowledge

1. The demo exposes 161 held-out samples from the 801-sample source data; it does not expose every source sample as an evaluation example.
2. `gene_<number>` identifiers have not been mapped to official gene symbols in the project.
3. The displayed holdout metrics are from this dataset; independent external validation is not shown.
4. Sample-specific SHAP is calculated only when requested and explains model output, not biological causation.
5. Only five cancer classes are supported.
6. This project is not a clinical diagnostic tool and should not guide patient care.

Acknowledging these limitations demonstrates sound scientific reasoning. Avoid describing feature rankings as proven biomarkers or model results as clinically validated.

## What to improve before the next presentation

1. Deploy the app and check Community Cloud build logs.
2. Verify upload validation with correctly formatted, malformed, missing-feature, and non-numeric CSV files.
3. Add a verified annotation mapping so feature IDs can be translated to gene symbols.
4. Validate on an independent cohort and assess potential batch effects.
5. Compare more models with repeated or nested validation.

## A confident closing answer

> Our system is a prototype for exploring how a small set of gene-expression features can classify samples among five cancer types and which features contribute to those predictions. Its held-out results are encouraging on this dataset, but the gene IDs are not yet biologically annotated and independent validation is still required before drawing biological or clinical conclusions.
