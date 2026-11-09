"""Build compact, reproducible artifacts for the hosted Streamlit demo."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "cancer_project" / "TCGA-PANCAN-HiSeq-801x20531"
FEATURE_COUNT = 20
RANDOM_STATE = 42


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    expression = pd.read_csv(DATA_DIR / "data.csv")
    labels = pd.read_csv(DATA_DIR / "labels.csv")
    if "Unnamed: 0" not in expression or "Unnamed: 0" not in labels:
        raise ValueError("Both source files must contain their sample identifier column.")

    expression = expression.rename(columns={"Unnamed: 0": "sample_id"})
    labels = labels.rename(columns={"Unnamed: 0": "sample_id"})
    if expression["sample_id"].duplicated().any() or labels["sample_id"].duplicated().any():
        raise ValueError("Sample identifiers must be unique before joining the data.")

    joined = expression.merge(
        labels[["sample_id", "Class"]],
        on="sample_id",
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    if len(joined) != len(expression) or len(joined) != len(labels):
        raise ValueError("Expression and label files do not contain the same sample IDs.")

    features = joined.drop(columns=["sample_id", "Class"])
    if not features.columns.is_unique or features.isna().any().any():
        raise ValueError("Feature columns must be unique and contain no missing values.")
    return joined, features


def main() -> None:
    joined, features = load_dataset()
    labels = joined["Class"]

    development_idx, test_idx = train_test_split(
        np.arange(len(joined)),
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    train_idx, validation_idx = train_test_split(
        development_idx,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=labels.iloc[development_idx],
    )

    selector = ExtraTreesClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    selector.fit(features.iloc[train_idx], labels.iloc[train_idx])
    ranked_indices = np.argsort(selector.feature_importances_)[::-1][:FEATURE_COUNT]
    feature_names = features.columns[ranked_indices].tolist()
    selected = features[feature_names]

    candidates: dict[str, Any] = {
        "Random forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Extra trees": ExtraTreesClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    candidate_scores: dict[str, dict[str, float]] = {}
    for name, candidate in candidates.items():
        candidate.fit(selected.iloc[train_idx], labels.iloc[train_idx])
        validation_prediction = candidate.predict(selected.iloc[validation_idx])
        candidate_scores[name] = {
            "accuracy": float(
                accuracy_score(labels.iloc[validation_idx], validation_prediction)
            ),
            "macro_f1": float(
                f1_score(labels.iloc[validation_idx], validation_prediction, average="macro")
            ),
        }

    best_name = max(
        candidate_scores,
        key=lambda name: (candidate_scores[name]["macro_f1"], candidate_scores[name]["accuracy"]),
    )
    model = candidates[best_name]
    model.fit(selected.iloc[development_idx], labels.iloc[development_idx])

    test_features = selected.iloc[test_idx]
    test_labels = labels.iloc[test_idx]
    test_prediction = model.predict(test_features)
    test_probabilities = model.predict_proba(test_features)
    classes = [str(label) for label in model.classes_]
    test_samples = joined.iloc[test_idx][["sample_id", "Class"]].copy()
    test_samples = test_samples.rename(columns={"Class": "true_class"}).reset_index(drop=True)
    test_samples = pd.concat(
        [test_samples, test_features.reset_index(drop=True)],
        axis=1,
    )

    feature_table = pd.DataFrame(
        {
            "feature": feature_names,
            "selection_importance": selector.feature_importances_[ranked_indices],
        }
    )
    bundle = {
        "model": model,
        "feature_names": feature_names,
        "classes": classes,
        "background": selected.iloc[development_idx[: min(64, len(development_idx))]]
        .to_numpy(),
        "selected_model": best_name,
    }
    metrics = {
        "source_sample_count": int(len(joined)),
        "source_feature_count": int(features.shape[1]),
        "selected_feature_count": int(len(feature_names)),
        "class_counts": {
            str(label): int(count) for label, count in labels.value_counts().sort_index().items()
        },
        "split": {
            "training_samples": int(len(train_idx)),
            "validation_samples": int(len(validation_idx)),
            "final_fit_samples": int(len(development_idx)),
            "test_samples": int(len(test_idx)),
            "random_state": RANDOM_STATE,
            "strategy": "stratified 60/20/20 train/validation/test",
        },
        "feature_selection": (
            "Top features ranked by an ExtraTrees model fit only on the training split."
        ),
        "model_comparison": {
            "selection_metric": "validation macro F1",
            "validation_scores": candidate_scores,
            "selected_model": best_name,
        },
        "held_out_test": {
            "accuracy": float(accuracy_score(test_labels, test_prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(test_labels, test_prediction)),
            "macro_f1": float(f1_score(test_labels, test_prediction, average="macro")),
            "classification_report": classification_report(
                test_labels,
                test_prediction,
                output_dict=True,
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(
                test_labels, test_prediction, labels=model.classes_
            ).tolist(),
            "class_order": classes,
            "probabilities": test_probabilities.tolist(),
        },
        "caveat": (
            "This is an internal holdout evaluation on one public dataset, not external "
            "or clinical validation."
        ),
    }

    joblib.dump(bundle, ROOT / "deployment_model.joblib", compress=3)
    test_samples.to_csv(ROOT / "deployment_demo.csv", index=False)
    feature_table.to_csv(ROOT / "deployment_features.csv", index=False)
    (ROOT / "deployment_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Built deployment artifacts for {len(joined)} aligned samples.")
    print(f"Selected model: {best_name}")
    print(f"Selected features: {', '.join(feature_names)}")
    print(
        "Held-out test: "
        f"accuracy={metrics['held_out_test']['accuracy']:.3f}, "
        f"macro F1={metrics['held_out_test']['macro_f1']:.3f}"
    )


if __name__ == "__main__":
    main()
