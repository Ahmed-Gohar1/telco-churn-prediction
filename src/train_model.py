"""Train RandomForest and save artifacts."""

import os
import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib


ROOT = os.path.dirname(os.path.dirname(__file__))
RAW = os.path.join(ROOT, "data", "telco_raw.csv")
OUT = os.path.join(ROOT, "artifacts")
os.makedirs(OUT, exist_ok=True)
MODEL_PATH = os.path.join(OUT, "best_model.joblib")
FEATURES_PATH = os.path.join(OUT, "feature_names.json")


def preprocess_df(df):
    """Return (X, y) after preprocessing."""
    # find target
    target = "Churn Label" if "Churn Label" in df.columns else ("Churn" if "Churn" in df.columns else None)
    if target is None:
        raise SystemExit("No churn target column found in the CSV")
    # drop ids/leak
    id_cols = [c for c in df.columns if c.lower().strip() in ("customer id", "customerid", "customer")]
    leak_cols = [c for c in df.columns if ("churn" in c.lower() and c != target) or c.lower().strip() == "customer status"]
    drop_cols = id_cols + leak_cols
    X = df.drop(columns=[c for c in drop_cols + [target] if c in df.columns]).copy()
    y = df[target].copy()
    # coerce numeric-like columns
    for col in ["TotalCharges", "Total Charges", "Total_Charges", "MonthlyCharge", "Monthly Charges", "Monthly Charge"]:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    # tenure features
    if "tenure" in X.columns:
        X["tenure_months"] = pd.to_numeric(X["tenure"], errors="coerce")
    elif "Tenure in Months" in X.columns:
        X["tenure_months"] = pd.to_numeric(X["Tenure in Months"], errors="coerce")
    if "tenure_months" in X.columns:
        bins = [-1, 1, 12, 24, 48, 1000]
        labels = ["0-1", "1-12", "12-24", "24-48", "48+"]
        X["tenure_bucket"] = pd.cut(X["tenure_months"], bins=bins, labels=labels)
    # binary flags
    if "Internet Service" in X.columns:
        X["has_internet"] = (~X["Internet Service"].isin(["No", "None", None])).astype(int)
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
    # drop high-cardinality
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    high_card = [c for c in cat_cols if X[c].nunique() > 40]
    X = X.drop(columns=high_card)
    # one-hot encode
    X = pd.get_dummies(X, drop_first=True)
    # convert y to 0/1
    if y.dtype == object:
        y = y.map({"Yes": 1, "No": 0, "yes": 1, "no": 0}).fillna(0).astype(int)
    # fill numeric NAs
    nums = X.select_dtypes(include=["number"]).columns
    for c in nums:
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].median())

    return X, y


def main():
    print("Loading:", RAW)
    df = pd.read_csv(RAW)
    print("Rows,cols:", df.shape)
    X, y = preprocess_df(df)
    print("Feature shape after preprocess:", X.shape)

    # split and train
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    print("Train score:", clf.score(X_train, y_train), "Val score:", clf.score(X_val, y_val))

    # save model and features
    joblib.dump(clf, MODEL_PATH)
    with open(FEATURES_PATH, "w", encoding="utf8") as f:
        json.dump(list(X.columns), f)
    print("Wrote model to", MODEL_PATH)
    print("Wrote feature names to", FEATURES_PATH)


if __name__ == "__main__":
    main()
