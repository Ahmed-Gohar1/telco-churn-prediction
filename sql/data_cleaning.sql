DROP TABLE IF EXISTS telco_cleaned_sql;

CREATE TABLE telco_cleaned_sql AS
SELECT
    
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
    
    CASE
        WHEN (COALESCE(NULLIF(TRIM(COALESCE("Tenure in Months", '')), ''), NULL) IS NOT NULL) THEN CAST(NULLIF(TRIM("Tenure in Months"), '') AS INTEGER)
        WHEN (COALESCE(NULLIF(TRIM(COALESCE(tenure, '')), ''), NULL) IS NOT NULL) THEN CAST(NULLIF(TRIM(tenure), '') AS INTEGER)
        ELSE NULL
    END AS tenure_months,
   
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

    "Offer",
    "Phone Service",
    "Avg Monthly Long Distance Charges",
    "Multiple Lines",
    "Internet Service",
    "Internet Type",
    "Avg Monthly GB Download",

    CASE
        WHEN TRIM(COALESCE("TotalCharges", "Total Charges", '')) = '' THEN NULL
        ELSE CAST(NULLIF(TRIM(COALESCE("TotalCharges", "Total Charges", '')), '') AS NUMERIC)
    END AS total_charges,

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

    CASE WHEN LOWER(TRIM(COALESCE("Internet Service", ''))) IN ('no','none','') THEN 0 ELSE 1 END AS has_internet,

    "Contract",
    "Paperless Billing",
    "Payment Method",

    COALESCE(CAST(NULLIF(TRIM(COALESCE("Monthly Charge", "Monthly Charges", "MonthlyCharge", '')), '') AS NUMERIC), 0) AS monthly_charge,
    COALESCE(CAST(NULLIF(TRIM(COALESCE("Total Refunds", '')), '') AS NUMERIC), 0) AS total_refunds,
    COALESCE(CAST(NULLIF(TRIM(COALESCE("Total Extra Data Charges", '')), '') AS NUMERIC), 0) AS total_extra_data_charges,
    COALESCE(CAST(NULLIF(TRIM(COALESCE("Total Long Distance Charges", '')), '') AS NUMERIC), 0) AS total_long_distance_charges,
    COALESCE(CAST(NULLIF(TRIM(COALESCE("Total Revenue", '')), '') AS NUMERIC), 0) AS total_revenue,

    "Satisfaction Score",
    "CLTV"

FROM staging_telco_raw;
