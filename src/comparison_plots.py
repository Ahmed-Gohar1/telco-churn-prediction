import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay
import joblib


ROOT = os.path.dirname(os.path.dirname(__file__))
ARTIFACTS = os.path.join(ROOT, 'artifacts')
os.makedirs(ARTIFACTS, exist_ok=True)

RAW = os.path.join(ROOT, 'data', 'telco_cleaned.csv')
MODEL_PATH = os.path.join(ARTIFACTS, 'best_model.joblib')
FEATURES_PATH = os.path.join(ARTIFACTS, 'feature_names.json')


RAW = os.path.join(ROOT, 'data', 'telco_raw.csv')
print('Loading raw data from', RAW)

def load_data(path):
    print('Loading cleaned data from', path)
    return pd.read_csv(path)


def build_dataset(df):
    # basic preprocessing: drop ids/leaks, map target, encode
    possible_targets = ['Churn Label', 'Churn']
    target = next((t for t in possible_targets if t in df.columns), None)
    if target is None:
        raise SystemExit('No target column found in cleaned CSV')

    drop_cols = [c for c in df.columns if c.lower().strip() in ('customer id','customerid','customer')]
    leak_cols = [c for c in df.columns if ('churn' in c.lower() and c != target) or c.lower().strip()=='customer status']
    drop_list = [c for c in ([target] + drop_cols + leak_cols) if c in df.columns]
    X = df.drop(columns=drop_list).copy()
    y = df[target].map({'Yes':1,'No':0,'yes':1,'no':0}).fillna(0).astype(int)

    # one-hot encode
    X_enc = pd.get_dummies(X, drop_first=True)

    # fill numeric NAs
    for c in X_enc.select_dtypes(include=['number']).columns:
        if X_enc[c].isna().any():
            X_enc[c] = X_enc[c].fillna(X_enc[c].median())

    return X_enc, y


def align_features(X, model):
    # align to saved feature list
    if os.path.exists(FEATURES_PATH):
        with open(FEATURES_PATH, 'r', encoding='utf8') as f:
            feats = json.load(f)
    else:
        feats = getattr(model, 'feature_names_in_', None)
        if feats is None:
            feats = list(X.columns)
    X_aligned = X.reindex(columns=feats, fill_value=0)
    return X_aligned


def plot_roc_pr(y_test, prob1, prob2, labels=('Model A','Model B')):
    fpr1, tpr1, _ = roc_curve(y_test, prob1)
    fpr2, tpr2, _ = roc_curve(y_test, prob2)
    auc1 = auc(fpr1, tpr1)
    auc2 = auc(fpr2, tpr2)
    plt.figure(figsize=(8,6))
    plt.plot(fpr1, tpr1, label=f'{labels[0]} (AUC={auc1:.3f})')
    plt.plot(fpr2, tpr2, label=f'{labels[1]} (AUC={auc2:.3f})')
    plt.plot([0,1],[0,1],'k--',alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC comparison')
    plt.legend()
    path = os.path.join(ARTIFACTS, 'roc_comparison.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print('Saved', path)

    # precision-recall
    p1, r1, _ = precision_recall_curve(y_test, prob1)
    p2, r2, _ = precision_recall_curve(y_test, prob2)
    ap1 = average_precision_score(y_test, prob1)
    ap2 = average_precision_score(y_test, prob2)
    plt.figure(figsize=(8,6))
    plt.plot(r1, p1, label=f'{labels[0]} (AP={ap1:.3f})')
    plt.plot(r2, p2, label=f'{labels[1]} (AP={ap2:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall comparison')
    plt.legend()
    path = os.path.join(ARTIFACTS, 'pr_comparison.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print('Saved', path)


def plot_confusion(y_test, pred1, pred2):
    fig, ax = plt.subplots(1,2,figsize=(12,5))
    ConfusionMatrixDisplay.from_predictions(y_test, pred1, ax=ax[0])
    ax[0].set_title('Model A CM')
    ConfusionMatrixDisplay.from_predictions(y_test, pred2, ax=ax[1])
    ax[1].set_title('Model B CM')
    path = os.path.join(ARTIFACTS, 'confusion_matrices.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print('Saved', path)


def plot_feature_importances(model, feat_names):
    if hasattr(model, 'feature_importances_'):
        fi = pd.Series(model.feature_importances_, index=feat_names).sort_values(ascending=False)
        plt.figure(figsize=(8,10))
        fi.head(25).sort_values().plot(kind='barh')
        path = os.path.join(ARTIFACTS, 'rf_feature_importances.png')
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        print('Saved', path)


def summarize(name, y_true, y_pred, y_prob):
    print(f'--- {name} ---')
    print('Accuracy:', accuracy_score(y_true, y_pred))
    print('Precision:', precision_score(y_true, y_pred))
    print('Recall:', recall_score(y_true, y_pred))
    print('F1:', f1_score(y_true, y_pred))
    print('ROC AUC:', auc(*roc_curve(y_true, y_prob)[:2]))


def main():
    df = load_data(RAW)
    X, y = build_dataset(df)

    # simple train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # load model
    if not os.path.exists(MODEL_PATH):
        raise SystemExit('Model not found at ' + MODEL_PATH)
    model = joblib.load(MODEL_PATH)

    X_test_aligned = align_features(X_test, model)
    X_train_aligned = align_features(X_train, model)

    # For LR vs RF, we'll train a simple LR here as model A and use loaded RF as model B
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_aligned, y_train)

    prob_lr = lr.predict_proba(X_test_aligned)[:,1]
    prob_rf = model.predict_proba(X_test_aligned)[:,1]
    pred_lr = (prob_lr >= 0.5).astype(int)
    pred_rf = (prob_rf >= 0.5).astype(int)

    plot_roc_pr(y_test, prob_lr, prob_rf, labels=('Logistic Regression','Random Forest'))
    plot_confusion(y_test, pred_lr, pred_rf)
    plot_feature_importances(model, X_train_aligned.columns.tolist())

    summarize('Logistic Regression', y_test, pred_lr, prob_lr)
    summarize('Random Forest', y_test, pred_rf, prob_rf)


if __name__ == '__main__':
    main()
