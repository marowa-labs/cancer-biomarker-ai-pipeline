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

## Key Features
- **High Accuracy**: Achieved 99.6% accuracy using Stratified K-Fold cross-validation.
- **Explainability**: Uses SHAP values to rank genes by their contribution to cancer classification.
- **Interactive Dashboard**: A professional Streamlit app for real-time prediction and biomarker visualization.

## Getting Started
1. Create a virtual environment: `python3 -m venv venv`
2. Activate it: `source venv/bin/activate`
3. Install dependencies: `pip install streamlit pandas joblib shap scikit-learn matplotlib`
4. Run the app: `streamlit run app.py`
# cancer-biomarker-ai-pipeline
