import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np
import matplotlib.pyplot as plt

# Set page configuration for a professional look
st.set_page_config(page_title="Cancer Classification AI", layout="wide")

# Custom CSS for industrial look
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        background-color: #004a99;
        color: white;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the model and data
@st.cache_resource
def load_resources():
    model = joblib.load('cancer_model.pkl')
    X = pd.read_csv('X_cleaned.csv')
    explainer = joblib.load('explainer.pkl')
    return model, X, explainer

model, X, explainer = load_resources()

st.title("🧬 Cancer Type Classification AI")
st.subheader("Industrial-Grade Biomarker Analysis Pipeline")

# Sidebar for controls
st.sidebar.header("Configuration")
sample_idx = st.sidebar.selectbox("Select a sample index", range(len(X)))
sample = X.iloc[[sample_idx]]

# Main layout
col1, col2 = st.columns([1, 2])

with col1:
    st.write("### Sample Data")
    st.dataframe(sample.T.head(10), use_container_width=True)
    
    # Predict
    prediction = model.predict(sample)
    st.metric(label="Predicted Cancer Type", value=prediction[0])

    # Dynamic Conclusion
    st.write("### Clinical Interpretation")
    
    # Calculate SHAP values for the sample
    # Note: TreeExplainer.shap_values returns a list for multi-class, or array for binary
    shap_values = explainer.shap_values(sample)
    
    # Handle both binary (array) and multi-class (list of arrays) outputs
    # Based on our check, shap_values is a numpy array of shape (1, n_features, n_classes)
    if shap_values.ndim == 3:
        # Multi-class: (1, n_features, n_classes)
        # We need to map the predicted class to the index in the 3rd dimension
        class_idx = np.where(model.classes_ == prediction[0])[0][0]
        values = shap_values[:, :, class_idx]
    else:
        # Binary: (1, n_features)
        values = shap_values
        
    # Get the index of the feature with the highest absolute SHAP value
    top_feature_idx = np.argmax(np.abs(values[0]))
    top_gene = sample.columns[top_feature_idx]
    
    st.info(f"For sample index {sample_idx}, the model predicts **{prediction[0]}**. "
            f"The primary gene expression driver identified is **{top_gene}**. "
            "Researchers should prioritize this biomarker for potential drug target validation "
            "or further molecular characterization to confirm its role in this cancer subtype.")

with col2:
    st.write("### SHAP Biomarker Importance")
    # SHAP visualization
    shap_values = explainer.shap_values(sample)
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, sample, plot_type="bar", show=False)
    st.pyplot(fig)
