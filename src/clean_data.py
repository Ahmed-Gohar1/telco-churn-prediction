import os
import pandas as pd


ROOT = os.path.dirname(os.path.dirname(__file__))
IN_PATH = os.path.join(ROOT, "data", "telco_raw.csv")
OUT_PATH = os.path.join(ROOT, "data", "telco_cleaned.csv")


def load_data(path):
    print("Loading:", path)
    return pd.read_csv(path)


def drop_ids_and_leaks(df):
    # Remove customer identifier columns and direct leakage columns that mention churn
    id_cols = [c for c in df.columns if c.lower().strip() in ("customer id", "customerid", "customer")]
    # Any column with 'churn' in its name (except the target) is considered leakage
    possible_targets = ["Churn Label", "Churn"]
    target = next((t for t in possible_targets if t in df.columns), None)
    leak_cols = [c for c in df.columns if ("churn" in c.lower() and c != target) or c.lower().strip() == "customer status"]
    to_drop = [c for c in ([target] + id_cols + leak_cols) if c in df.columns]
    if to_drop:
        print("Dropping columns (target/ids/leak):", to_drop)
        return df.drop(columns=to_drop)
    return df


def coerce_totals(df):
    # Convert common total charge column names to numeric and treat blanks as NA
    for col in ["TotalCharges", "Total Charges", "Total_Charges"]:
        if col in df.columns:
            print(f"Coercing {col} -> numeric (blank -> NA)")
            df[col] = df[col].astype(str).str.strip().replace("", pd.NA)
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def add_tenure(df):
    # Normalize tenure column and add simple buckets that are easy to explain
    if "tenure" in df.columns:
        df["tenure_months"] = pd.to_numeric(df["tenure"], errors="coerce")
    elif "Tenure in Months" in df.columns:
        df["tenure_months"] = pd.to_numeric(df["Tenure in Months"], errors="coerce")

    if "tenure_months" in df.columns:
        bins = [-1, 1, 12, 24, 48, 1000]
        labels = ["0-1", "1-12", "12-24", "24-48", "48+"]
        df["tenure_bucket"] = pd.cut(df["tenure_months"], bins=bins, labels=labels)
    return df


def add_simple_flags(df):
    # Create a has_internet flag and map common yes/no service cols to 0/1
    if "Internet Service" in df.columns:
        df["has_internet"] = (~df["Internet Service"].isin(["No", "None", None])).astype(int)

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
    map_values = {"Yes": 1, "No": 0, "yes": 1, "no": 0, "No internet service": 0, "No phone service": 0}
    for col in binary_cols:
        if col in df.columns:
            new_col = col + "_bin"
            df[new_col] = df[col].map(map_values).fillna(0).astype(int)
    return df


def fill_numeric_with_median(df):
    # Fill numeric missing values conservatively with the median
    nums = df.select_dtypes(include=["number"]).columns
    for c in nums:
        if df[c].isna().any():
            med = df[c].median()
            df[c] = df[c].fillna(med)
    return df


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df = load_data(IN_PATH)
    print("Shape before:", df.shape)
    df = drop_ids_and_leaks(df)
    df = coerce_totals(df)
    df = add_tenure(df)
    df = add_simple_flags(df)
    df = fill_numeric_with_median(df)
    print("Shape after:", df.shape)
    df.to_csv(OUT_PATH, index=False)
    print("Wrote:", OUT_PATH)


if __name__ == "__main__":
    main()
