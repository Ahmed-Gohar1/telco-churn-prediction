import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    classification_report,
    ConfusionMatrixDisplay,
)
import joblib

sns.set(style='whitegrid')

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(ROOT, 'data', 'telco_cleaned.csv')
OUT = os.path.join(ROOT, 'artifacts')
os.makedirs(OUT, exist_ok=True)

print('Loading cleaned data from', DATA)
df = pd.read_csv(DATA)

# load raw target if cleaned file dropped it
raw = pd.read_csv(os.path.join(ROOT, 'data', 'telco_raw.csv'))
target = 'Churn Label' if 'Churn Label' in raw.columns else ('Churn' if 'Churn' in raw.columns else None)
if target is None:
    raise SystemExit('No churn target found in raw data')

# merge target into cleaned df if needed
if target in df.columns:
    y = df[target]
    X = df.drop(columns=[target])
else:
    # if cleaned removed target, use raw's target aligned by original row order
    y = raw[target]
    X = df.copy()

# normalize target to binary 0/1
def _to_binary_series(s):
    # handle various string/number encodings
    try:
        arr = pd.Series(s)
    except Exception:
        arr = pd.Series(list(s))
    if arr.dtype == object or arr.dtype.name == 'category':
        mapping = {'yes': 1, 'y': 1, 'true': 1, 't': 1, '1': 1,
                   'no': 0, 'n': 0, 'false': 0, 'f': 0, '0': 0}
        return arr.astype(str).str.strip().str.lower().map(mapping).astype(float).fillna(arr)
    if np.issubdtype(arr.dtype, np.number):
        unique = np.unique(arr[~pd.isna(arr)])
        if set(unique.tolist()) <= {0, 1}:
            return arr.astype(int)
        return (arr != 0).astype(int)
    return arr

# apply mapping
y = _to_binary_series(y)

X = X.copy()
# numeric coercion
for col in ['total_charges', 'Monthly Charge', 'MonthlyCharge', 'Monthly Charges', 'MonthlyCharge']:
    if col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
# encode categoricals conservatively
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
high_card = [c for c in cat_cols if X[c].nunique() > 40]
X_enc = X.drop(columns=high_card)
X_enc = pd.get_dummies(X_enc, drop_first=True)
# impute numerics
num_cols = X_enc.select_dtypes(include=['number']).columns.tolist()
imp = SimpleImputer(strategy='median')
X_enc[num_cols] = imp.fit_transform(X_enc[num_cols])
# split and scale
X_train, X_test, y_train, y_test = train_test_split(X_enc, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# train models
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
# save
joblib.dump(lr, os.path.join(OUT, 'lr_model.joblib'))
joblib.dump(rf, os.path.join(OUT, 'rf_model.joblib'))
joblib.dump(scaler, os.path.join(OUT, 'scaler.joblib'))

# ROC
plt.figure(figsize=(8, 6))
lr_probs = lr.predict_proba(X_test_scaled)[:, 1]
rf_probs = rf.predict_proba(X_test)[:, 1]
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
plt.plot(lr_fpr, lr_tpr, label=f'LogisticRegression (AUC={roc_auc_score(y_test, lr_probs):.3f})')
plt.plot(rf_fpr, rf_tpr, label=f'RandomForest (AUC={roc_auc_score(y_test, rf_probs):.3f})')
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')
roc_path = os.path.join(OUT, 'model_comparison_roc.png')
plt.savefig(roc_path, bbox_inches='tight', dpi=150)
print('Wrote', roc_path)
plt.close()

# Precision-recall
plt.figure(figsize=(8, 6))
lr_prec, lr_rec, _ = precision_recall_curve(y_test, lr_probs)
rf_prec, rf_rec, _ = precision_recall_curve(y_test, rf_probs)
plt.plot(lr_rec, lr_prec, label=f'LogisticRegression (AP={average_precision_score(y_test, lr_probs):.3f})')
plt.plot(rf_rec, rf_prec, label=f'RandomForest (AP={average_precision_score(y_test, rf_probs):.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve Comparison')
plt.legend(loc='lower left')
pr_path = os.path.join(OUT, 'model_comparison_pr.png')
plt.savefig(pr_path, bbox_inches='tight', dpi=150)
print('Wrote', pr_path)
plt.close()

# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
lr_pred = lr.predict(X_test_scaled)
rf_pred = rf.predict(X_test)
ConfusionMatrixDisplay.from_predictions(y_test, lr_pred, ax=axes[0], cmap='Blues', display_labels=[0,1])
axes[0].set_title('LogisticRegression')
ConfusionMatrixDisplay.from_predictions(y_test, rf_pred, ax=axes[1], cmap='Blues', display_labels=[0,1])
axes[1].set_title('RandomForest')
cm_path = os.path.join(OUT, 'model_comparison_confusion.png')
plt.savefig(cm_path, bbox_inches='tight', dpi=150)
print('Wrote', cm_path)
plt.close()

# Feature importances and LR coefficients
if hasattr(rf, 'feature_importances_'):
    fi = rf.feature_importances_
    idx = np.argsort(fi)[::-1][:20]
    names = np.array(X_train.columns)[idx]
    vals = fi[idx]
    plt.figure(figsize=(8, 6))
    sns.barplot(x=vals, y=names, palette='viridis')
    plt.title('Top 20 RandomForest feature importances')
    plt.xlabel('Importance')
    fi_path = os.path.join(OUT, 'rf_feature_importances.png')
    plt.savefig(fi_path, bbox_inches='tight', dpi=150)
    print('Wrote', fi_path)
    plt.close()

if hasattr(lr, 'coef_'):
    coefs = lr.coef_[0]
    idx = np.argsort(np.abs(coefs))[::-1][:20]
    names = np.array(X_train.columns)[idx]
    vals = coefs[idx]
    plt.figure(figsize=(8, 6))
    sns.barplot(x=vals, y=names, palette='coolwarm')
    plt.title('Top 20 LogisticRegression coefficients (by absolute value)')
    plt.xlabel('Coefficient')
    coef_path = os.path.join(OUT, 'lr_top_coefficients.png')
    plt.savefig(coef_path, bbox_inches='tight', dpi=150)
    print('Wrote', coef_path)
    plt.close()


print('\nClassification report (LogisticRegression):')
print(classification_report(y_test, lr_pred))
print('\nClassification report (RandomForest):')
print(classification_report(y_test, rf_pred))
print('\nAll visuals saved to artifacts/')
