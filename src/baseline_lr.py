# baseline_lr.py -- Logistic Regression classifier for smishing detection
# Dataset      : Dataset_5971.csv (5,949 clean rows after dedup)
#                Binary label: 0=ham (4,834)  |  1=malicious/smishing (1,115)
# Run with     : python src/baseline_lr.py

import pandas as pd
import numpy as np
import joblib, os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix, classification_report)

# ── 1. Load preprocessed splits ───────────────────────────────────────────────
# These files are created by preprocess.py.
# Columns available: text, label_binary, label_orig, has_url, has_phone, has_email
print('Loading data...')
train = pd.read_csv('data/train.csv')
val   = pd.read_csv('data/val.csv')

print(f'Train: {len(train):,} rows  |  Val: {len(val):,} rows')
print(f'Train label distribution: {train["label_binary"].value_counts().to_dict()}')

X_train, y_train = train['text'], train['label_binary']
X_val,   y_val   = val['text'],   val['label_binary']

# ── 2. TF-IDF: convert SMS text to numbers ────────────────────────────────────
# max_features=10000 : keep the 10,000 most informative words/phrases
# ngram_range=(1,2)  : consider single words AND two-word phrases
#                      e.g. "click here" scores higher than "click" alone
# sublinear_tf=True  : dampens very high word counts (log scaling)
print('\nFitting TF-IDF vectorizer...')
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}'   # includes single characters like 'u'
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf   = vectorizer.transform(X_val)

print(f'TF-IDF matrix shape (train): {X_train_tfidf.shape}')

# ── 3. Train Logistic Regression ──────────────────────────────────────────────
# class_weight='balanced': automatically adjusts for the 81/19 class imbalance
#   (4,834 ham vs 1,115 malicious). Without this, the model would be biased
#   toward always predicting ham and still achieve ~81% accuracy.
# C=1.0              : regularisation strength (1.0 is standard starting point)
# max_iter=1000      : enough iterations for convergence on TF-IDF sparse data
print('\nTraining Logistic Regression...')
lr_model = LogisticRegression(
    C=1.0,
    class_weight='balanced',
    max_iter=1000,
    solver='lbfgs',
    random_state=42
)
lr_model.fit(X_train_tfidf, y_train)
print('Logistic Regression training complete.')

# ── 4. Evaluate on validation set ────────────────────────────────────────────
y_pred      = lr_model.predict(X_val_tfidf)
y_prob      = lr_model.predict_proba(X_val_tfidf)[:, 1]  # probability of malicious

acc   = accuracy_score(y_val, y_pred)
prec  = precision_score(y_val, y_pred, average='binary', zero_division=0)
rec   = recall_score(y_val, y_pred, average='binary', zero_division=0)
f1    = f1_score(y_val, y_pred, average='binary', zero_division=0)
cm    = confusion_matrix(y_val, y_pred)
tn, fp, fn, tp = cm.ravel()
fpr   = fp / (fp + tn)   # false positive rate: ham wrongly flagged as malicious

print(f'\n===== Logistic Regression Results =====')
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

# ── 5. Top 20 most phishing-associated words ──────────────────────────────────
# Positive coefficients = words that push prediction toward malicious (class 1)
feature_names = vectorizer.get_feature_names_out()
coefs         = lr_model.coef_[0]
top20_idx     = coefs.argsort()[-20:][::-1]
print('Top 20 words most associated with smishing/malicious:')
for i in top20_idx:
    print(f'  {feature_names[i]:<25} weight: {coefs[i]:.4f}')

# ── 6. Save model, vectorizer, and results ────────────────────────────────────
os.makedirs('models',  exist_ok=True)
os.makedirs('outputs', exist_ok=True)

joblib.dump(lr_model,   'models/lr_model.pkl')
joblib.dump(vectorizer, 'models/lr_vectorizer.pkl')

pd.DataFrame([{
    'model'    : 'Logistic Regression',
    'accuracy' : acc,
    'precision': prec,
    'recall'   : rec,
    'f1'       : f1,
    'fpr'      : fpr,
    'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
}]).to_csv('outputs/lr_results.csv', index=False)

print('\nModel saved  : models/lr_model.pkl')
print('Results saved: outputs/lr_results.csv')
print('Done.')
