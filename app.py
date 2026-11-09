from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "deployment_model.joblib"
DEMO_PATH = ROOT / "deployment_demo.csv"
FEATURES_PATH = ROOT / "deployment_features.csv"
METRICS_PATH = ROOT / "deployment_metrics.json"

CANCER_NAMES = {
    "BRCA": "Breast invasive carcinoma",
    "COAD": "Colon adenocarcinoma",
    "KIRC": "Kidney renal clear cell carcinoma",
    "LUAD": "Lung adenocarcinoma",
    "PRAD": "Prostate adenocarcinoma",
}

st.set_page_config(
    page_title="Cancer expression research demo",
    page_icon=":material/biotech:",
    layout="wide",
)


@st.cache_resource
def load_model_bundle() -> dict:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Deployment model not found: {MODEL_PATH.name}")
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_demo_samples() -> pd.DataFrame:
    if not DEMO_PATH.is_file():
        raise FileNotFoundError(f"Demo samples not found: {DEMO_PATH.name}")
    return pd.read_csv(DEMO_PATH)


@st.cache_data
def load_model_metrics() -> dict:
    if not METRICS_PATH.is_file():
        raise FileNotFoundError(f"Model evaluation report not found: {METRICS_PATH.name}")
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_feature_ranking() -> pd.DataFrame:
    if not FEATURES_PATH.is_file():
        raise FileNotFoundError(f"Feature ranking not found: {FEATURES_PATH.name}")
    return pd.read_csv(FEATURES_PATH)


def class_description(code: str) -> str:
    return f"{code} — {CANCER_NAMES.get(code, 'Cancer class')}"


def prediction_table(model: object, samples: pd.DataFrame) -> pd.DataFrame:
    probabilities = model.predict_proba(samples)
    class_codes = model.classes_
    predictions = model.predict(samples)
    result = pd.DataFrame(
        {
            "Predicted class": [class_description(str(value)) for value in predictions],
            "Model score": probabilities.max(axis=1),
        }
    )
    for index, code in enumerate(class_codes):
        result[f"P({code})"] = probabilities[:, index]
    return result


def local_shap_values(
    model: object,
    sample: pd.DataFrame,
    predicted_class: str,
) -> np.ndarray:
    explainer = shap.TreeExplainer(model)
    raw_values = explainer.shap_values(sample, check_additivity=False)
    class_index = list(model.classes_).index(predicted_class)

    if isinstance(raw_values, list):
        return np.asarray(raw_values[class_index])[0]

    values = np.asarray(raw_values)
    if values.ndim == 3 and values.shape[-1] == len(model.classes_):
        return values[0, :, class_index]
    if values.ndim == 3 and values.shape[0] == len(model.classes_):
        return values[class_index, 0, :]
    if values.ndim == 2:
        return values[0]
    raise ValueError(f"Unexpected SHAP output shape: {values.shape}")


bundle = load_model_bundle()
model = bundle["model"]
feature_names = bundle["feature_names"]
demo = load_demo_samples()
metrics = load_model_metrics()
feature_ranking = load_feature_ranking()

required_demo_columns = {"sample_id", "true_class", *feature_names}
missing_demo_columns = required_demo_columns.difference(demo.columns)
if missing_demo_columns:
    raise ValueError(f"Demo data is missing columns: {', '.join(sorted(missing_demo_columns))}")
if list(model.classes_) != bundle["classes"]:
    raise ValueError("Saved model classes do not match the deployment metadata.")

st.title("Cancer type classification")
st.caption(
    "An interactive research demo using gene-expression profiles from a public "
    "TCGA Pan-Cancer dataset."
)
st.badge("Research prototype", icon=":material/science:", color="orange")
st.warning(
    "For research and education only. This model is not clinically validated and "
    "must not be used to diagnose, screen, or guide treatment.",
    icon=":material/health_and_safety:",
)

test_metrics = metrics["held_out_test"]
with st.container(horizontal=True):
    st.metric("Source samples", f"{metrics['source_sample_count']:,}", border=True)
    st.metric("Source gene features", metrics["source_feature_count"], border=True)
    st.metric("Selected model features", len(feature_names), border=True)
    st.metric("Cancer classes", len(bundle["classes"]), border=True)
    st.metric("Held-out accuracy", f"{test_metrics['accuracy']:.1%}", border=True)
    st.metric("Held-out macro F1", f"{test_metrics['macro_f1']:.1%}", border=True)

with st.expander("Cancer class abbreviations"):
    st.dataframe(
        pd.DataFrame(
            [
                {"Code": code, "Cancer type": CANCER_NAMES.get(code, "Cancer class")}
                for code in bundle["classes"]
            ]
        ),
        width="stretch",
        hide_index=True,
    )

st.caption(
    f"Model: {bundle['selected_model']} using {len(feature_names)} selected features. "
    f"Metrics come from a stratified test split of {metrics['split']['test_samples']} samples; "
    "they are not external or clinical validation."
)

view = st.segmented_control(
    "Choose a view",
    options=["Explore held-out samples", "Upload expression data", "Model evaluation"],
    default="Explore held-out samples",
    key="main_view",
)

if view == "Explore held-out samples":
    sample_ids = demo["sample_id"].astype(str).tolist()
    selected_id = st.selectbox("Select a held-out sample", sample_ids)
    selected_row = demo.loc[demo["sample_id"].astype(str) == selected_id].iloc[0]
    sample = demo.loc[[selected_row.name], feature_names]
    result = prediction_table(model, sample)
    predicted_code = str(model.predict(sample)[0])
    true_code = str(selected_row["true_class"])
    score = float(result.loc[0, "Model score"])

    st.caption(
        f"Sample ID: `{selected_id}` · Observed class: **{class_description(true_code)}**"
    )
    columns = st.columns(3)
    columns[0].metric("Model prediction", predicted_code, border=True)
    columns[1].metric("Model score", f"{score:.1%}", border=True)
    columns[2].metric(
        "Holdout result",
        "Correct" if predicted_code == true_code else "Incorrect",
        border=True,
    )
    st.caption(
        "The model score is the model's class probability, not a measure of clinical certainty."
    )

    probability_frame = pd.DataFrame(
        {
            "Cancer class": [str(code) for code in model.classes_],
            "Probability": model.predict_proba(sample)[0],
        }
    ).sort_values("Probability", ascending=False)

    probability_column, feature_column = st.columns([1, 1])
    with probability_column:
        with st.container(border=True):
            st.subheader("Class probabilities")
            st.bar_chart(
                probability_frame,
                x="Cancer class",
                y="Probability",
                horizontal=True,
            )
            st.caption(
                "Model probabilities across the five classes. See the class-abbreviation "
                "guide above for full names."
            )

    with feature_column:
        with st.container(border=True):
            st.subheader("Selected gene features")
            st.dataframe(
                sample.T.rename(columns={sample.index[0]: "Expression"}),
                width="stretch",
            )
            st.caption(
                "Feature IDs are dataset column names, not annotated gene symbols."
            )

    with st.container(border=True):
        st.subheader("Model-level feature ranking")
        ranking = feature_ranking.sort_values(
            "selection_importance", ascending=False
        ).head(15)
        chart = ranking.rename(
            columns={
                "feature": "Gene feature",
                "selection_importance": "Selection importance",
            }
        )
        st.bar_chart(
            chart,
            x="Gene feature",
            y="Selection importance",
            horizontal=True,
        )
        st.caption(
            "These global scores were used to select features from the training split. "
            "They are not sample-specific SHAP explanations and do not imply causation."
        )

    if st.button("Explain this prediction with SHAP", icon=":material/insights:"):
        try:
            local_values = local_shap_values(model, sample, predicted_code)
            local_frame = pd.DataFrame(
                {"Gene feature": feature_names, "SHAP value": local_values}
            )
            local_frame = local_frame.iloc[
                np.argsort(np.abs(local_values))[::-1][:15]
            ]
            st.subheader(f"Sample-level SHAP explanation for {predicted_code}")
            st.bar_chart(
                local_frame,
                x="Gene feature",
                y="SHAP value",
                horizontal=True,
            )
            st.caption(
                "Positive and negative values show how features move the model output "
                "for the predicted class. They describe model behavior, not biological "
                "causality or clinical importance."
            )
        except Exception as error:
            st.error(
                f"SHAP could not explain this prediction: "
                f"{type(error).__name__}: {error}"
            )

elif view == "Upload expression data":
    st.subheader("Predict from a CSV")
    st.write(
        "Upload one or more samples with the 20 required `gene_<number>` columns. "
        "Additional metadata columns are ignored."
    )
    st.caption(
        "Input values must use the same expression scale and preprocessing as the "
        "TCGA training data. The app cannot verify that for you."
    )
    uploaded_file = st.file_uploader(
        "Expression data CSV",
        type=["csv"],
        max_upload_size=10,
    )

    if uploaded_file is not None:
        try:
            uploaded = pd.read_csv(uploaded_file)
        except (
            UnicodeDecodeError,
            pd.errors.EmptyDataError,
            pd.errors.ParserError,
            OSError,
        ) as error:
            st.error(f"Could not read the CSV: {error}")
        else:
            missing_features = [name for name in feature_names if name not in uploaded]
            if missing_features:
                st.error(
                    "The CSV is missing required model features: "
                    + ", ".join(missing_features)
                )
            elif uploaded.empty:
                st.error("The uploaded CSV contains no sample rows.")
            elif len(uploaded) > 200:
                st.error("Upload at most 200 samples at a time.")
            else:
                input_features = uploaded.loc[:, feature_names].apply(
                    pd.to_numeric, errors="coerce"
                )
                if input_features.isna().any().any() or not np.isfinite(
                    input_features.to_numpy()
                ).all():
                    st.error("Required gene features must contain finite numeric values.")
                else:
                    predictions = prediction_table(model, input_features)
                    if "sample_id" in uploaded.columns:
                        predictions.insert(0, "Sample ID", uploaded["sample_id"].astype(str))
                    else:
                        predictions.insert(
                            0,
                            "Sample ID",
                            [f"upload_{index + 1}" for index in range(len(uploaded))],
                        )
                    st.dataframe(predictions, width="stretch", hide_index=True)
                    st.download_button(
                        "Download predictions",
                        predictions.to_csv(index=False).encode("utf-8"),
                        file_name="cancer_predictions.csv",
                        mime="text/csv",
                        icon=":material/download:",
                    )

elif view == "Model evaluation":
    st.subheader("How the model was evaluated")
    st.write(
        "Samples were split by class into training, validation, and held-out test sets. "
        "Feature ranking used only the training subset; model selection used validation "
        "macro F1; the test set was reserved for the final report."
    )

    comparison = pd.DataFrame(metrics["model_comparison"]["validation_scores"]).T
    comparison.index.name = "Candidate model"
    st.markdown("**Validation comparison**")
    st.dataframe(comparison, width="stretch")
    st.caption(
        f"Selected using {metrics['model_comparison']['selection_metric']}: "
        f"{metrics['model_comparison']['selected_model']}."
    )

    left, right = st.columns(2)
    with left:
        st.markdown("**Held-out classification report**")
        report = pd.DataFrame(test_metrics["classification_report"]).T
        st.dataframe(report, width="stretch")

    with right:
        st.markdown("**Held-out confusion matrix**")
        confusion = pd.DataFrame(
            test_metrics["confusion_matrix"],
            index=metrics["held_out_test"]["class_order"],
            columns=metrics["held_out_test"]["class_order"],
        )
        st.dataframe(confusion, width="stretch")
        st.caption("Rows are observed classes; columns are predicted classes.")

    st.warning(metrics["caveat"], icon=":material/info:")

st.caption(
    "Cancer class abbreviations follow the labels in the supplied dataset. "
    "Gene feature IDs require a separate verified annotation table to map them to symbols."
)
