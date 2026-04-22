import os
import json
import streamlit as st
import pandas as pd
import joblib


st.title("Telco Churn Demo")
ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(ROOT, "artifacts", "best_model.joblib")
FEATURES_PATH = os.path.join(ROOT, "artifacts", "feature_names.json")


@st.cache_resource
def load_model_and_features():
    model = joblib.load(MODEL_PATH)
    features = None
    if os.path.exists(FEATURES_PATH):
        with open(FEATURES_PATH, "r", encoding="utf8") as f:
            features = json.load(f)
    return model, features


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()

    # 1) make totals numeric
    for col in ["TotalCharges", "Total Charges", "Total_Charges"]:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    # 2) simple tenure handling
    if "tenure" in X.columns:
        X["tenure_months"] = pd.to_numeric(X["tenure"], errors="coerce")
    elif "Tenure in Months" in X.columns:
        X["tenure_months"] = pd.to_numeric(X["Tenure in Months"], errors="coerce")
    if "tenure_months" in X.columns:
        X["tenure_bucket"] = pd.cut(X["tenure_months"], bins=[-1, 1, 12, 24, 48, 1000], labels=["0-1", "1-12", "12-24", "24-48", "48+"])

    # 3) flag whether the customer has internet
    if "Internet Service" in X.columns:
        X["has_internet"] = (~X["Internet Service"].isin(["No", "None", None])).astype(int)

    # 4) map common Yes/No columns to 0/1
    binary_map = {"Yes": 1, "No": 0, "yes": 1, "no": 0, "No internet service": 0, "No phone service": 0}
    binary_cols = [
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
    ]
    for col in binary_cols:
        if col in X.columns:
            X[col + "_bin"] = X[col].map(binary_map).fillna(0).astype(int)

    # 5) quick one-hot for remaining categoricals (keeps code short)
    X_enc = pd.get_dummies(X, drop_first=True)

    # 6) fill numeric NA with median to avoid model errors
    for c in X_enc.select_dtypes(include=["number"]).columns:
        if X_enc[c].isna().any():
            X_enc[c] = X_enc[c].fillna(X_enc[c].median())

    return X_enc


st.sidebar.markdown("Upload a CSV with the same columns as the project telco CSV")
uploaded = st.sidebar.file_uploader("CSV file", type=["csv"])

if not os.path.exists(MODEL_PATH):
    st.warning(f"Model not found at {MODEL_PATH}. Run the training script to create it.")
else:
    model, feature_names = load_model_and_features()

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        X = preprocess(df)

        # fall back to the prepared columns from this run.
        if feature_names is None:
            model_features = list(X.columns)
        else:
            model_features = feature_names

        X_aligned = X.reindex(columns=model_features, fill_value=0)

        preds = model.predict(X_aligned)
        probs = model.predict_proba(X_aligned)[:, 1]

        out = df.copy()
        out["churn_pred"] = preds
        out["churn_prob"] = probs

        st.dataframe(out.head())
        st.download_button("Download predictions (CSV)", out.to_csv(index=False), file_name="preds.csv")
    else:
        st.info("Upload a CSV to get predictions.")
