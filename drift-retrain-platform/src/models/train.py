"""
Training utilities for the orchestration pipeline.
Automatically selects the best model.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pandas as pd


def train_baseline_model(X_ref, y_ref, metadata=None, model_config=None):
    unique_classes = pd.Series(y_ref).nunique() if hasattr(y_ref, "nunique") else len(set(y_ref))

    stratify_target = None
    if unique_classes > 1:
        class_counts = pd.Series(y_ref).value_counts()
        if (class_counts >= 2).all() and len(y_ref) >= 5:
            stratify_target = y_ref

    if len(X_ref) >= 5 and unique_classes > 1:
        X_train, X_val, y_train, y_val = train_test_split(
            X_ref,
            y_ref,
            test_size=0.2,
            random_state=42,
            stratify=stratify_target
        )
    else:
        X_train, X_val, y_train, y_val = X_ref, X_ref, y_ref, y_ref

    candidate_models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        )
    }

    if unique_classes > 1:
        candidate_models["Logistic Regression"] = LogisticRegression(
            max_iter=500,
            random_state=42
        )

    best_model = None
    best_score = -1
    best_name = ""

    print("\n========== Training Candidate Models ==========\n")

    for model_name, model in candidate_models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_val)

        if unique_classes > 1:
            score = f1_score(y_val, predictions, zero_division=0)
        else:
            score = 1.0

        print(f"{model_name:<25} F1 Score : {score:.4f}")

        if score > best_score:
            best_score = score
            best_model = model
            best_name = model_name

    print("\n===============================================")
    print(f"Selected Model : {best_name}")
    print(f"Best F1 Score  : {best_score:.4f}")
    print("===============================================\n")

    return best_model


def retrain_model(X_ref, y_ref, X_batch, y_batch=None, metadata=None, model_config=None):
    if y_batch is None or (isinstance(y_batch, (pd.Series, pd.DataFrame)) and y_batch.empty):
        X_new = X_ref
        y_new = y_ref
    else:
        X_new = pd.concat([X_ref, X_batch], ignore_index=True)
        y_new = pd.concat([y_ref, y_batch], ignore_index=True)

    return train_baseline_model(
        X_ref=X_new,
        y_ref=y_new,
        metadata=metadata,
        model_config=model_config
    )
