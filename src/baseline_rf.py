# baseline_rf.py -- Random Forest classifier for smishing detection
# Dataset      : Dataset_5971.csv (5,949 clean rows after dedup)
#                Binary label: 0=ham (4,834)  |  1=malicious/smishing (1,115)
# Run with     : python src/baseline_rf.py
# Expected time: 2 to 5 minutes

import pandas as pd
import numpy as np
import joblib, os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

# ── 1. Load preprocessed splits ───────────────────────────────────────────────
print('Loading data...')
train = pd.read_csv('data/train.csv')
val   = pd.read_csv('data/val.csv')

print(f'Train: {len(train):,} rows  |  Val: {len(val):,} rows')
print(f'Train label distribution: {train["label_binary"].value_counts().to_dict()}')

X_train, y_train = train['text'], train['label_binary']
X_val,   y_val   = val['text'],   val['label_binary']

# ── 2. Build extra handcrafted features from URL/PHONE/EMAIL columns ──────────
# Your dataset explicitly flags whether each message contains a URL, phone
# number, or email address. These are strong smishing indicators and give
# Random Forest additional signal beyond just word counts.
train_extra = train[['has_url', 'has_phone', 'has_email']].values
val_extra   = val[['has_url', 'has_phone', 'has_email']].values

# ── 3. TF-IDF vectorisation ───────────────────────────────────────────────────
print('\nFitting TF-IDF vectorizer...')
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}'
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf   = vectorizer.transform(X_val)

# ── 4. Combine TF-IDF + handcrafted features ──────────────────────────────────
# Stack TF-IDF sparse matrix with the 3 binary feature columns (URL/phone/email)
# scipy.sparse is required because TF-IDF output is a sparse matrix
from scipy.sparse import hstack, csr_matrix

X_train_combined = hstack([X_train_tfidf, csr_matrix(train_extra)])
X_val_combined   = hstack([X_val_tfidf,   csr_matrix(val_extra)])

print(f'Combined feature matrix shape (train): {X_train_combined.shape}')
# Shape will be (n_samples, 10003) -- 10000 TF-IDF + 3 handcrafted features

# ── 5. Train Random Forest ────────────────────────────────────────────────────
# n_estimators=200  : build 200 decision trees and let them vote
# class_weight='balanced' : compensates for 81/19 ham/malicious imbalance
# n_jobs=-1         : use all available CPU cores (speeds up training)
print('\nTraining Random Forest (200 trees)...')
print('This takes 2 to 5 minutes. Please wait.')

rf_model = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    max_depth=None,       # trees grow fully unless constrained
    min_samples_leaf=2,   # prevents overfitting on tiny leaf nodes
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_combined, y_train)
print('Random Forest training complete.')

# ── 6. Evaluate on validation set ────────────────────────────────────────────
y_pred = rf_model.predict(X_val_combined)

acc   = accuracy_score(y_val, y_pred)
prec  = precision_score(y_val, y_pred, average='binary', zero_division=0)
rec   = recall_score(y_val, y_pred, average='binary', zero_division=0)
f1    = f1_score(y_val, y_pred, average='binary', zero_division=0)
tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
fpr   = fp / (fp + tn)

print(f'\n===== Random Forest Results =====')
print(f'Accuracy            : {acc:.4f}')
print(f'Precision           : {prec:.4f}')
print(f'Recall              : {rec:.4f}')
print(f'F1-Score            : {f1:.4f}')
print(f'False Positive Rate : {fpr:.4f}')
print(f'\nConfusion Matrix:')
print(f'  TN={tn}  FP={fp}')
print(f'  FN={fn}  TP={tp}')
print(f'\nPer-class report:')
print(classification_report(y_val, y_pred, target_names=['Ham (0)', 'Malicious (1)']))

# ── 7. Feature importances (top 20) ──────────────────────────────────────────
# Shows which words and features drove predictions most.
# Compare these against what your LLM agent flags in its CoT reasoning.
feature_names = list(vectorizer.get_feature_names_out()) + ['has_url', 'has_phone', 'has_email']
importances   = rf_model.feature_importances_
top20_idx     = importances.argsort()[-20:][::-1]

print('\nTop 20 most important features:')
for i in top20_idx:
    print(f'  {feature_names[i]:<25} importance: {importances[i]:.6f}')

# ── 8. Save model, vectorizer, and results ────────────────────────────────────
os.makedirs('models',  exist_ok=True)
os.makedirs('outputs', exist_ok=True)

joblib.dump(rf_model,   'models/rf_model.pkl')
joblib.dump(vectorizer, 'models/rf_vectorizer.pkl')

pd.DataFrame([{
    'model'    : 'Random Forest',
    'accuracy' : acc,
    'precision': prec,
    'recall'   : rec,
    'f1'       : f1,
    'fpr'      : fpr,
    'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
}]).to_csv('outputs/rf_results.csv', index=False)

print('\nModel saved  : models/rf_model.pkl')
print('Results saved: outputs/rf_results.csv')
print('Done.')
