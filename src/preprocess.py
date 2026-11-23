# preprocess.py  -- Dataset: SMS_PHISHING_DATASET_FOR_MACHINE_LEARNING_AND_PATTERN_RECOGNITION
# Source file   : Dataset_5971.csv  (5,971 rows | columns: LABEL, TEXT, URL, EMAIL, PHONE)
# Run with      : python src/preprocess.py

import pandas as pd
import unicodedata
import os
from sklearn.model_selection import train_test_split

# ── 1. Load ──────────────────────────────────────────────────────────────────
# latin-1 encoding is required -- the file contains non-UTF-8 characters
# such as  (corrupted pound sign) in some smishing messages.
RAW_PATH = 'data/Dataset_5971.csv'

print(f'Loading dataset from {RAW_PATH} ...')
df = pd.read_csv(RAW_PATH, encoding='latin-1')
print(f'Loaded {len(df):,} rows | Columns: {df.columns.tolist()}')

# ── 2. Rename columns to lowercase for consistency ────────────────────────────
df.columns = [c.strip().upper() for c in df.columns]   # ensure clean uppercase
# Keep TEXT as the message column; LABEL as the label column

# ── 3. Normalise labels (fix mixed capitalisation) ────────────────────────────
# Raw values found: 'ham', 'Smishing', 'smishing', 'spam', 'Spam'
print('\nRaw label distribution:')
print(df['LABEL'].value_counts().to_string())

df['LABEL'] = df['LABEL'].str.strip().str.lower()

print('\nNormalised label distribution:')
print(df['LABEL'].value_counts().to_string())

# ── 4. Fix encoding artefacts in TEXT ─────────────────────────────────────────
# Unicode NFKC normalisation resolves garbled characters like  and Â£
def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = unicodedata.normalize('NFKC', text)
    text = text.strip()
    return text

df['TEXT'] = df['TEXT'].apply(clean_text)

# ── 5. Drop empty / null rows ─────────────────────────────────────────────────
before = len(df)
df = df[df['TEXT'].str.len() > 0]
df = df.dropna(subset=['TEXT', 'LABEL'])
print(f'\nDropped {before - len(df)} empty/null rows.')

# ── 6. Remove duplicate messages ──────────────────────────────────────────────
before = len(df)
df = df.drop_duplicates(subset='TEXT')
print(f'Removed {before - len(df)} duplicate messages. Remaining: {len(df):,}')

# ── 7. Create binary label ────────────────────────────────────────────────────
# Strategy: smishing + spam = malicious (1) | ham = legitimate (0)
# This maps directly onto your binary smishing detection framing.
# "spam" is included because it represents unsolicited/malicious content
# that shares characteristics with smishing in this dataset.
df['label_binary'] = df['LABEL'].apply(lambda x: 1 if x in ['smishing', 'spam'] else 0)

print('\nBinary label distribution:')
print(df['label_binary'].value_counts().to_string())
print(f"  Class ratio (ham:malicious) = "
      f"{(df['label_binary']==0).sum()} : {(df['label_binary']==1).sum()}")

# ── 8. Select and rename columns for downstream scripts ───────────────────────
# 'text'         -- the SMS message (used by ALL classifiers)
# 'label_binary' -- 0=ham, 1=malicious (smishing/spam)
# 'label_orig'   -- original normalised label (for reference/error analysis)
# 'has_url'      -- 1 if URL present (extra feature for RF/XGBoost)
# 'has_phone'    -- 1 if phone number present (extra feature for RF/XGBoost)
# 'has_email'    -- 1 if email address present (extra feature for RF/XGBoost)

df['text']       = df['TEXT']
df['label_orig'] = df['LABEL']
df['has_url']    = (df['URL'].str.strip().str.lower() == 'yes').astype(int)
df['has_phone']  = (df['PHONE'].str.strip().str.lower() == 'yes').astype(int)
df['has_email']  = (df['EMAIL'].str.strip().str.lower() == 'yes').astype(int)

keep_cols = ['text', 'label_binary', 'label_orig', 'has_url', 'has_phone', 'has_email']
df = df[keep_cols].reset_index(drop=True)

# ── 9. Stratified 70 / 15 / 15 split ─────────────────────────────────────────
# stratify=df['label_binary'] ensures both classes are proportionally
# represented in every split -- essential given the 81/19 class imbalance.
train_df, temp_df = train_test_split(
    df, test_size=0.30, stratify=df['label_binary'], random_state=42)

val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df['label_binary'], random_state=42)

print(f'\nSplit sizes:')
print(f'  Train : {len(train_df):>5,}  '
      f'(ham={( train_df["label_binary"]==0).sum()}, '
      f'malicious={(train_df["label_binary"]==1).sum()})')
print(f'  Val   : {len(val_df):>5,}  '
      f'(ham={( val_df["label_binary"]==0).sum()}, '
      f'malicious={(val_df["label_binary"]==1).sum()})')
print(f'  Test  : {len(test_df):>5,}  '
      f'(ham={( test_df["label_binary"]==0).sum()}, '
      f'malicious={(test_df["label_binary"]==1).sum()})')

# ── 10. Save ──────────────────────────────────────────────────────────────────
os.makedirs('data', exist_ok=True)
train_df.to_csv('data/train.csv', index=False)
val_df.to_csv('data/val.csv',     index=False)
test_df.to_csv('data/test.csv',   index=False)

print('\nSaved:')
print('  data/train.csv')
print('  data/val.csv')
print('  data/test.csv  <-- DO NOT open or use until final evaluation')
print('\nPreprocessing complete.')
