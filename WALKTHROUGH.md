# Project Walkthrough: Cancer Type Classification AI

## Introduction
This document provides a detailed walkthrough of the Cancer Type Classification AI pipeline. This project was designed to bridge the gap between raw genomic data and actionable clinical insights.

## Pipeline Steps

### 1. Data Acquisition & Preprocessing
We utilize the **TCGA Pan-Cancer RNA-Seq dataset** (TCGA-PANCAN-HiSeq-801x20531).
- **Source**: [The Cancer Genome Atlas (TCGA)](https://www.cancer.gov/ccg/research/genome-sequencing/tcga)
- **Data Portal**: [National Cancer Institute (NCI) GDC Data Portal](https://portal.gdc.cancer.gov/)
- **Details**: This is a **Pan-Cancer** dataset, meaning it aggregates RNA-Seq gene expression data for 801 samples across multiple cancer types, with 20,531 genes measured per sample. This allows our model to perform multi-class classification, distinguishing between different cancer types based on their unique gene expression signatures.
- **Preprocessing**: The raw data undergoes cleaning, including the removal of non-numeric identifiers and normalization to ensure model robustness.

### 2. Model Training
A `RandomForestClassifier` is trained on the preprocessed gene expression data. We employ `StratifiedKFold` cross-validation to ensure the model generalizes well across different cancer types and to prevent data leakage.

### 3. Explainable AI (XAI)
To move beyond "black-box" predictions, we implement `SHAP` (SHapley Additive exPlanations). This allows us to quantify the contribution of each gene to the model's prediction, effectively turning the model into a biomarker discovery tool.

### 4. Deployment
The final model, explainer, and processed data are integrated into a `Streamlit` dashboard. This provides an interactive interface for researchers to explore predictions and visualize biomarker importance.
### Dashboard Guide
- **Select a sample index**: This allows you to choose a specific patient sample from the dataset. By selecting an index, you isolate that patient's gene expression profile to analyze how the model arrived at its prediction.
- **Sample Data**: This displays the gene expression levels for the top 10 genes (biomarkers) for the selected patient.
- **SHAP Biomarker Importance**: This graph uses SHAP to show which genes had the most influence on the model's prediction for this specific patient.
    - **Interpreting the graph**: The length of each bar represents the magnitude of the feature's impact on the model's output. A longer bar means that specific gene was a major "decider" for the model.
    - **Colors**: The 5 colors represent the different cancer classes the model was trained to distinguish.
    - **Higher vs. Lower**: A higher (longer) bar means that gene is a stronger influencer for that specific cancer class. If a gene has a long bar in one color but a short bar in others, it is a highly specific biomarker for that cancer type.
- **Zero Expression**: You may notice some genes (like `gene_0`, `gene_5`, `gene_8`, `gene_9`) have zero expression. This is common in genomic datasets, indicating that these genes are not expressed in the samples you are analyzing.
## Why This Pipeline Matters
- **Clinical Relevance**: By identifying top-contributing genes, this pipeline helps researchers focus on potential therapeutic targets.
- **Transparency**: The use of SHAP ensures that every prediction is backed by biological evidence, increasing trust in the AI's output.
- **Scalability**: The modular design allows for easy integration of new datasets or more complex classification models.

## Industrial Standard
Unlike simple classification projects, this pipeline emphasizes:
- **Robust Validation**: Rigorous cross-validation to ensure reliability.
- **Interpretability**: Prioritizing biological insights over raw accuracy.
- **Professional UI**: A clean, dashboard-driven interface designed for end-users (researchers/clinicians).
