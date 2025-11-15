-- data_cleaning.sql
-- SQL-based data cleaning for the Telco dataset.
-- Assumptions:
-- 1) Raw data is loaded into a staging table named `staging_telco_raw` with column names matching the CSV headers.
-- 2) This script creates a cleaned table `telco_cleaned_sql` (replace names as needed) that mirrors the transformations
--    used in the Python cleaning script: drop identifiers and leakage columns, coerce numeric columns, create tenure
--    buckets and binary flags for common service features.
-- 3) This SQL is written in ANSI-style SQL and uses portable expressions where possible. You may need minor tweaks for
--    your target dialect (Postgres, MySQL, SQLite, etc.).

-- Drop the output table if it exists (dialect-specific; adjust if necessary)
DROP TABLE IF EXISTS telco_cleaned_sql;

-- Create cleaned table
CREATE TABLE telco_cleaned_sql AS
SELECT
    -- keep a selection of original columns (excluding identifiers and leakage columns)
    "Gender",
    "Age",
    "Under 30",
    "Senior Citizen",
    "Married",
    "Dependents",
    "Number of Dependents",
    "Country",
    "State",
    "City",
    "Zip Code",
    "Latitude",
    "Longitude",
    "Population",
    "Quarter",
    "Referred a Friend",
    "Number of Referrals",
    -- normalize tenure: prefer 'Tenure in Months' otherwise fallback to 'tenure'
    CASE
        WHEN (COALESCE(NULLIF(TRIM(COALESCE("Tenure in Months", '')), ''), NULL) IS NOT NULL) THEN CAST(NULLIF(TRIM("Tenure in Months"), '') AS INTEGER)
        WHEN (COALESCE(NULLIF(TRIM(COALESCE(tenure, '')), ''), NULL) IS NOT NULL) THEN CAST(NULLIF(TRIM(tenure), '') AS INTEGER)
        ELSE NULL
    END AS tenure_months,
    -- tenure bucket
    CASE
        WHEN (CASE
                WHEN (COALESCE(NULLIF(TRIM(COALESCE("Tenure in Months", '')), ''), NULL) IS NOT NULL) THEN CAST(NULLIF(TRIM("Tenure in Months"), '') AS INTEGER)
                WHEN (COALESCE(NULLIF(TRIM(COALESCE(tenure, '')), ''), NULL) IS NOT NULL) THEN CAST(NULLIF(TRIM(tenure), '') AS INTEGER)
                ELSE NULL
             END) IS NULL THEN NULL
        WHEN (CASE
                WHEN (COALESCE(NULLIF(TRIM(COALESCE("Tenure in Months", '')), ''), NULL) IS NOT NULL) THEN CAST(NULLIF(TRIM("Tenure in Months"), '') AS INTEGER)
                WHEN (COALESCE(NULLIF(TRIM(COALESCE(tenure, '')), ''), NULL) IS NOT NULL) THEN CAST(NULLIF(TRIM(tenure), '') AS INTEGER)
                ELSE NULL
             END) <= 1 THEN '0-1'
        WHEN (CASE
                WHEN (COALESCE(NULLIF(TRIM(COALESCE("Tenure in Months", '')), ''), NULL) IS NOT NULL) THEN CAST(NULLIF(TRIM("Tenure in Months"), '') AS INTEGER)
                WHEN (COALESCE(NULLIF(TRIM(COALESCE(tenure, '')), ''), NULL) IS NOT NULL) THEN CAST(NULLIF(TRIM(tenure), '') AS INTEGER)
                ELSE NULL
             END) <= 12 THEN '1-12'
        WHEN (CASE
                WHEN (COALESCE(NULLIF(TRIM(COALESCE("Tenure in Months", '')), ''), NULL) IS NOT NULL) THEN CAST(NULLIF(TRIM("Tenure in Months"), '') AS INTEGER)
                WHEN (COALESCE(NULLIF(TRIM(COALESCE(tenure, '')), ''), NULL) IS NOT NULL) THEN CAST(NULLIF(TRIM(tenure), '') AS INTEGER)
                ELSE NULL
             END) <= 24 THEN '12-24'
        WHEN (CASE
                WHEN (COALESCE(NULLIF(TRIM(COALESCE("Tenure in Months", '')), ''), NULL) IS NOT NULL) THEN CAST(NULLIF(TRIM("Tenure in Months"), '') AS INTEGER)
                WHEN (COALESCE(NULLIF(TRIM(COALESCE(tenure, '')), ''), NULL) IS NOT NULL) THEN CAST(NULLIF(TRIM(tenure), '') AS INTEGER)
                ELSE NULL
             END) <= 48 THEN '24-48'
        ELSE '48+'
    END AS tenure_bucket,

    -- Offer and service columns (kept raw for potential one-hot encoding later)
    "Offer",
    "Phone Service",
    "Avg Monthly Long Distance Charges",
    "Multiple Lines",
    "Internet Service",
    "Internet Type",
    "Avg Monthly GB Download",

    -- Coerce TotalCharges / Total Charges to numeric (NULL if empty)
    CASE
        WHEN TRIM(COALESCE("TotalCharges", "Total Charges", '')) = '' THEN NULL
        ELSE CAST(NULLIF(TRIM(COALESCE("TotalCharges", "Total Charges", '')), '') AS NUMERIC)
    END AS total_charges,

    -- Binary mappings: normalized to 1/0
    CASE WHEN LOWER(TRIM(COALESCE("Online Security", ''))) = 'yes' THEN 1 ELSE 0 END AS online_security_bin,
    CASE WHEN LOWER(TRIM(COALESCE("Online Backup", ''))) = 'yes' THEN 1 ELSE 0 END AS online_backup_bin,
    CASE WHEN LOWER(TRIM(COALESCE("Device Protection Plan", ''))) = 'yes' THEN 1 ELSE 0 END AS device_protection_bin,
    CASE WHEN LOWER(TRIM(COALESCE("Premium Tech Support", ''))) = 'yes' THEN 1 ELSE 0 END AS premium_tech_support_bin,
    CASE WHEN LOWER(TRIM(COALESCE("Streaming TV", ''))) = 'yes' THEN 1 ELSE 0 END AS streaming_tv_bin,
    CASE WHEN LOWER(TRIM(COALESCE("Streaming Movies", ''))) = 'yes' THEN 1 ELSE 0 END AS streaming_movies_bin,
    CASE WHEN LOWER(TRIM(COALESCE("Streaming Music", ''))) = 'yes' THEN 1 ELSE 0 END AS streaming_music_bin,
    CASE WHEN LOWER(TRIM(COALESCE("Unlimited Data", ''))) = 'yes' THEN 1 ELSE 0 END AS unlimited_data_bin,
    CASE WHEN LOWER(TRIM(COALESCE("Phone Service", ''))) = 'yes' THEN 1 ELSE 0 END AS phone_service_bin,
    CASE WHEN LOWER(TRIM(COALESCE("Multiple Lines", ''))) = 'yes' THEN 1 ELSE 0 END AS multiple_lines_bin,

    -- has_internet flag
    CASE WHEN LOWER(TRIM(COALESCE("Internet Service", ''))) IN ('no','none','') THEN 0 ELSE 1 END AS has_internet,

    -- Keep payment and billing info (raw)
    "Contract",
    "Paperless Billing",
    "Payment Method",

    -- Financials
    COALESCE(CAST(NULLIF(TRIM(COALESCE("Monthly Charge", "Monthly Charges", "MonthlyCharge", '')), '') AS NUMERIC), 0) AS monthly_charge,
    COALESCE(CAST(NULLIF(TRIM(COALESCE("Total Refunds", '')), '') AS NUMERIC), 0) AS total_refunds,
    COALESCE(CAST(NULLIF(TRIM(COALESCE("Total Extra Data Charges", '')), '') AS NUMERIC), 0) AS total_extra_data_charges,
    COALESCE(CAST(NULLIF(TRIM(COALESCE("Total Long Distance Charges", '')), '') AS NUMERIC), 0) AS total_long_distance_charges,
    COALESCE(CAST(NULLIF(TRIM(COALESCE("Total Revenue", '')), '') AS NUMERIC), 0) AS total_revenue,

    -- Retain satisfaction and CLTV if present
    "Satisfaction Score",
    "CLTV"

FROM staging_telco_raw;

-- Notes / next steps:
-- - If you are using Postgres, you can convert NUMERIC casts to DOUBLE PRECISION or DECIMAL(12,2) for performance.
-- - Indexes: consider adding an index on City, State, or tenure_bucket if you run analytics on those fields.
-- - This script purposefully keeps categorical columns (Offer, Payment Method, Internet Type, City) as raw text
--   so you can perform controlled encoding (one-hot, target encoding) downstream. If you prefer one-step one-hot
--   encoding in SQL, add CASE/COUNT columns or use a pivot technique (dialect-specific).

-- Example indexes (uncomment if desired in Postgres):
-- CREATE INDEX idx_telco_tenure_bucket ON telco_cleaned_sql (tenure_bucket);
-- CREATE INDEX idx_telco_city ON telco_cleaned_sql (City);
