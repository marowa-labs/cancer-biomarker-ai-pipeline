# Cancer Type Classification AI Pipeline

## Overview
This project implements an industrial-standard bioinformatics pipeline for classifying cancer types based on gene expression data. It leverages machine learning for classification and Explainable AI (XAI) to identify key biological biomarkers, providing actionable insights for researchers.

## Table of Contents
1. [Project Structure](#project-structure)
2. [Pipeline Workflow](#pipeline-workflow)
3. [Key Features](#key-features)
4. [Getting Started](#getting-started)

## Project Structure
```text
cancer_project/
├── app.py                # Streamlit dashboard
├── cancer_classification.ipynb # Core ML pipeline
├── cancer_model.pkl      # Trained Random Forest model
├── explainer.pkl         # SHAP explainer
├── X_cleaned.csv         # Preprocessed gene expression data
├── y.csv                 # Target labels
└── top_20_biomarkers.csv # Identified top 20 genes
```

## Pipeline Workflow
```mermaid
graph LR
    A[Raw Data] --> B[Preprocessing]
    B --> C[Random Forest Training]
    C --> D[Cross-Validation]
    D --> E[SHAP Interpretation]
    E --> F[Biomarker Extraction]
    F --> G[Streamlit Dashboard]
```

### Visual Artifacts & Dashboard Previews
Here are some visual insights and captures from the pipeline and data portal:

![GDC Data Portal](data/images/GDC-Data-Portal.png)
*Figure 1: GDC Data Portal gene expression data source.*

![Index Visualization](data/images/indexs.png)
*Figure 2: Index / Data distribution overview.*

![Dashboard Preview 1](data/images/Screenshot%202026-08-08%20203353.png)
*Figure 3: Streamlit dashboard workflow preview.*

![Dashboard Preview 2](data/images/Screenshot%202026-08-08%20203425.png)
*Figure 4: Detailed biomarker and prediction view.*

## Key Features
- **High Accuracy**: Achieved 99.6% accuracy using Stratified K-Fold cross-validation.
- **Explainability**: Uses SHAP values to rank genes by their contribution to cancer classification.
- **Interactive Dashboard**: A professional Streamlit app for real-time prediction and biomarker visualization.

## Getting Started
1. Create a virtual environment: `python3 -m venv venv`
2. Activate it: `source venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Run the app: `streamlit run app.py`
# cancer-biomarker-ai-pipeline
