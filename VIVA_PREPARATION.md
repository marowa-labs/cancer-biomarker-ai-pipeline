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

When you select a held-out sample, the model uses that sample's 20 expression values to generate a prediction. Selecting a different sample does not retrain the model:

- **Class probabilities** are recalculated for the selected sample, with one score for each of the five classes. The class with the highest score is the prediction. These are model outputs, not clinical certainty or patient risk.
- **Selected gene features** shows that sample's expression values for the same 20 model inputs. The feature names stay the same between samples; their expression values can differ. These values are not importance scores.
- **Model-level feature ranking** stays the same when the sample changes. It shows global Extra Trees importance scores, calculated using only the training split, for the top 15 features. These scores describe the feature-selection stage across training data; they do not explain an individual prediction.
- **Sample-level SHAP explanation** is calculated for the selected sample only after clicking “Explain this prediction with SHAP.” It displays up to 15 features ranked by the magnitude of their SHAP values for the predicted class. Positive and negative values show the direction in which each feature moves that class's model output relative to the explanation's baseline.

SHAP and feature-importance values describe model behavior; neither proves biological causation or clinical biomarker utility. After changing samples, click the SHAP button again to calculate an explanation for the new selection.

**Presentation-ready explanation:**

> “I select a held-out sample, and the trained model processes its 20 gene-expression features to produce a score for each of the five cancer classes. The highest score is the predicted class. The app also shows the known class for this held-out sample, so we can compare the prediction with its dataset label. The feature table shows the expression values used as input. The model-level ranking is global, based on the training data, so it does not change with the selected sample. SHAP is different: it explains this sample's prediction by showing which features moved the model output toward or away from the predicted class. These are explanations of the model, not proof of biological causes or clinical certainty.”

“Held-out” means the sample was reserved from model training for evaluation within this dataset. It does not mean that it came from an independent hospital, study, or population.

### How do I explain the model evaluation tables?

The evaluation view separates model selection from final testing. Feature selection used the training split, Random Forest and Extra Trees were compared on the validation split, and the held-out test split was reserved for the final performance report. The chosen model is Extra Trees, based on validation macro F1; the held-out test results were not used to choose between candidates.

- **Validation comparison:** Each row is a candidate algorithm, and the columns show validation accuracy and macro F1. Accuracy is the fraction of validation samples classified correctly. Macro F1 calculates F1 for each class and averages those class scores equally, so large classes do not dominate the summary. Extra Trees had the higher validation macro F1 (about 0.990 versus 0.984 for Random Forest), so it was selected.
- **Held-out classification report:** Each cancer-class row reports precision, recall, F1-score, and support. Precision answers, “Of the samples predicted as this class, what fraction truly belonged to it?” Recall answers, “Of the true samples in this class, what fraction did the model find?” F1 balances precision and recall. Support is the number of held-out samples whose observed label is that class. The summary rows give aggregate results; the `accuracy` row is overall accuracy.
- **Held-out confusion matrix:** Rows are the observed (true) classes and columns are the model's predicted classes. Counts on the diagonal are correct classifications. Off-diagonal counts show which true class was mistaken for which predicted class. The class order is shown in the table headers; these are counts, not percentages.

For example, the held-out report has 161 samples and overall accuracy of about 98.1%. This describes performance on one split from this dataset only. It is not external validation, and it does not establish accuracy on new populations or clinical samples.

### How could the takeaways help researchers?

If adapted and rigorously validated for research use, a tool like this could help researchers explore expression patterns, compare a sample with the five classes supported by the model, and flag predictions that differ from known labels for further investigation. Feature rankings and SHAP explanations could help prioritize patterns for literature review, independent-dataset analysis, or laboratory follow-up. These outputs are research leads, not proof that a gene causes cancer.

The current app is only an exploratory prototype. It supports five classes, its gene IDs have not been mapped to verified gene symbols, and its reported performance comes from one held-out split of this dataset. Before relying on it in real-world research, it would need verified biological annotation, independent-cohort validation, and testing across datasets and laboratories. It should not be used to diagnose patients or guide treatment.

**Presentation-ready answer:**

> “In a real research setting, this could be a hypothesis-generation and exploration tool. It may help researchers inspect expression patterns, find samples for follow-up, and prioritize features for further study. However, our current prototype has only been evaluated on one split of one dataset, supports five cancer classes, and does not yet map its gene IDs to verified gene names. We would need independent validation and biological follow-up before drawing conclusions. It is not a diagnostic tool and should not guide patient treatment.”

### Was the app deployed?

Yes. The research demo is publicly hosted on Streamlit Community Cloud at [cancer-biomarker-ai-pipeline.streamlit.app](https://cancer-biomarker-ai-pipeline.streamlit.app/). It is an interactive research prototype, not a clinical diagnostic service.

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

1. Verify upload validation with correctly formatted, malformed, missing-feature, and non-numeric CSV files.
2. Add a verified annotation mapping so feature IDs can be translated to gene symbols.
3. Validate on an independent cohort and assess potential batch effects.
4. Compare more models with repeated or nested validation.

## A confident closing answer

> Our system is a prototype for exploring how a small set of gene-expression features can classify samples among five cancer types and which features contribute to those predictions. Its held-out results are encouraging on this dataset, but the gene IDs are not yet biologically annotated and independent validation is still required before drawing biological or clinical conclusions.
