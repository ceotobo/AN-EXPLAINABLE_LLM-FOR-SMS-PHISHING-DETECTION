# baseline_xgb.py -- XGBoost classifier for smishing detection
# Dataset      : Dataset_5971.csv (5,949 clean rows after dedup)
#                Binary label: 0=ham (4,834)  |  1=malicious/smishing (1,115)
# Run with     : python src/baseline_xgb.py
# Expected time: 3 to 10 minutes

import pandas as pd
import numpy as np
import joblib, os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from scipy.sparse import hstack, csr_matrix
from xgboost import XGBClassifier

# ── 1. Load preprocessed splits ───────────────────────────────────────────────
print('Loading data...')
train = pd.read_csv('data/train.csv')
val   = pd.read_csv('data/val.csv')

print(f'Train: {len(train):,} rows  |  Val: {len(val):,} rows')
print(f'Train label distribution: {train["label_binary"].value_counts().to_dict()}')

X_train, y_train = train['text'], train['label_binary']
X_val,   y_val   = val['text'],   val['label_binary']

# ── 2. Handcrafted features (URL / PHONE / EMAIL flags) ───────────────────────
# Your dataset provides these pre-extracted binary columns.
# XGBoost benefits particularly from these structured features
# because they are explicit phishing indicators.
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
X_train_combined = hstack([X_train_tfidf, csr_matrix(train_extra)])
X_val_combined   = hstack([X_val_tfidf,   csr_matrix(val_extra)])

print(f'Combined feature matrix shape (train): {X_train_combined.shape}')

# ── 5. Compute class imbalance ratio for scale_pos_weight ────────────────────
# XGBoost uses scale_pos_weight instead of class_weight='balanced'.
# Formula: number of negative (ham) samples / number of positive (malicious) samples
# Your dataset: 4,834 ham / 1,115 malicious ≈ 4.33
# This tells XGBoost to treat each malicious example as ~4.33 ham examples.
neg   = int((y_train == 0).sum())
pos   = int((y_train == 1).sum())
scale = neg / pos

print(f'\nClass imbalance ratio (ham / malicious): {scale:.2f}')
print(f'  ham={neg}, malicious={pos}')
print(f'  scale_pos_weight will be set to {scale:.2f}')

# ── 6. Train XGBoost ──────────────────────────────────────────────────────────
# n_estimators=300   : build 300 boosting rounds
# learning_rate=0.1  : step size per round (lower = slower but more precise)
# max_depth=6        : maximum tree depth
# subsample=0.8      : use 80% of training data per round (reduces overfitting)
# colsample_bytree=0.8: use 80% of features per tree
# eval_metric='logloss': monitors log loss on validation set during training
# verbose=50         : print progress every 50 rounds
print('\nTraining XGBoost (300 rounds)...')
print('Progress prints every 50 rounds. logloss should decrease each time.')

xgb_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale,
    eval_metric='logloss',
    random_state=42,
    tree_method='hist',   # fastest method for sparse TF-IDF matrices
    n_jobs=-1
)

xgb_model.fit(
    X_train_combined, y_train,
    eval_set=[(X_val_combined, y_val)],
    verbose=50
)
print('XGBoost training complete.')

# ── 7. Evaluate on validation set ────────────────────────────────────────────
y_pred = xgb_model.predict(X_val_combined)
y_prob = xgb_model.predict_proba(X_val_combined)[:, 1]

acc   = accuracy_score(y_val, y_pred)
prec  = precision_score(y_val, y_pred, average='binary', zero_division=0)
rec   = recall_score(y_val, y_pred, average='binary', zero_division=0)
f1    = f1_score(y_val, y_pred, average='binary', zero_division=0)
tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
fpr   = fp / (fp + tn)

print(f'\n===== XGBoost Results =====')
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

# ── 8. Save model, vectorizer, and results ────────────────────────────────────
os.makedirs('models',  exist_ok=True)
os.makedirs('outputs', exist_ok=True)

joblib.dump(xgb_model,  'models/xgb_model.pkl')
joblib.dump(vectorizer, 'models/xgb_vectorizer.pkl')

pd.DataFrame([{
    'model'    : 'XGBoost',
    'accuracy' : acc,
    'precision': prec,
    'recall'   : rec,
    'f1'       : f1,
    'fpr'      : fpr,
    'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
}]).to_csv('outputs/xgb_results.csv', index=False)

print('\nModel saved  : models/xgb_model.pkl')
print('Results saved: outputs/xgb_results.csv')
print('Done.')
