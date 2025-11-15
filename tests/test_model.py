"""Simple smoke test to ensure the saved model can make a prediction.

This test uses a tiny, clear preprocessing pipeline that mirrors the
training script. The goal is readability rather than full parity with the
notebook; keeping the steps explicit helps interns understand what's required
to call a saved scikit-learn model.
"""

import joblib
import pandas as pd


def test_model_predicts_one_row():
    # Load the saved model. Training script writes to artifacts/best_model.joblib
    model = joblib.load("artifacts/best_model.joblib")

    # Read a small sample from the raw CSV and do a minimal, explicit
    # preprocessing so the model receives a DataFrame with matching columns.
    df = pd.read_csv("data/telco_raw.csv")

    # Drop ids and obvious leakage columns (the same idea used during training)
    target = "Churn Label" if "Churn Label" in df.columns else ("Churn" if "Churn" in df.columns else None)
    drop_cols = [c for c in df.columns if c.lower().strip() in ("customer id", "customerid", "customer")]
    leak_cols = [c for c in df.columns if ("churn" in c.lower() and c != target) or c.lower().strip() == "customer status"]
    drop_list = [c for c in ([target] + drop_cols + leak_cols) if c in df.columns]
    X = df.drop(columns=drop_list)

    # Add a couple of small derived columns used by the training script
    if "Tenure in Months" in X.columns:
        X["tenure_months"] = X["Tenure in Months"]
        X["tenure_bucket"] = pd.cut(X["tenure_months"], bins=[-1, 1, 12, 24, 48, 1000], labels=["0-1", "1-12", "12-24", "24-48", "48+"])
    if "Internet Service" in X.columns:
        X["has_internet"] = (~X["Internet Service"].isin(["No", "None", None])).astype(int)

    # Map a few Yes/No columns to 0/1
    binary_map = {"Yes": 1, "No": 0, "yes": 1, "no": 0}
    for col in [
        "Online Security",
        "Online Backup",
        "Device Protection Plan",
        "Premium Tech Support",
        "Streaming TV",
        "Streaming Movies",
        "Streaming Music",
        "Unlimited Data",
        "Phone Service",
        "Multiple Lines",
    ]:
        if col in X.columns:
            X[col + "_bin"] = X[col].map(binary_map).fillna(0).astype(int)

    # One-hot encode remaining categoricals and select a single row for a smoke test
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    high_card = [c for c in cat_cols if X[c].nunique() > 40]
    X_enc = X.drop(columns=high_card)
    X_enc = pd.get_dummies(X_enc, drop_first=True)

    sample = X_enc.iloc[:1]
    preds = model.predict(sample)
    assert len(preds) == 1
