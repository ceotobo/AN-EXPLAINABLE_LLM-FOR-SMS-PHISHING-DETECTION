# baseline_distilbert.py -- Fine-tune DistilBERT + compute SHAP values
# Dataset      : Dataset_5971.csv (5,949 clean rows after dedup)
#                Binary label: 0=ham (4,834)  |  1=malicious/smishing (1,115)
# Run with     : python src/baseline_distilbert.py
# Expected time: 1 to 2 hours (training) + 10 to 20 minutes (SHAP)
#
# BEFORE RUNNING: install SHAP if not already installed:
#   pip install shap==0.45.0

import pandas as pd
import numpy as np
import os, pickle, torch
from transformers import (DistilBertTokenizer,
                          DistilBertForSequenceClassification,
                          Trainer, TrainingArguments, pipeline)
from torch.utils.data import Dataset
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
import shap

# ── 1. Load preprocessed splits ───────────────────────────────────────────────
# DistilBERT does NOT use TF-IDF. It reads raw text directly.
# It also does NOT use has_url / has_phone / has_email features --
# it learns context from the full message text instead.
print('Loading data...')
train_df = pd.read_csv('data/train.csv')
val_df   = pd.read_csv('data/val.csv')

print(f'Train: {len(train_df):,} rows  |  Val: {len(val_df):,} rows')
print(f'Train label distribution: {train_df["label_binary"].value_counts().to_dict()}')

# ── 2. Load DistilBERT tokenizer ──────────────────────────────────────────────
# distilbert-base-uncased: converts text to lowercase before tokenising.
# This is appropriate for SMS text where capitalisation is inconsistent.
print('\nLoading DistilBERT tokenizer...')
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# ── 3. Dataset class ──────────────────────────────────────────────────────────
# PyTorch Dataset wrapper that tokenises each message and pairs it with its label.
# max_length=128: SMS messages are short (avg ~15 words); 128 tokens is sufficient.
#                 Truncation is enabled for the rare long messages.
class SMSDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

print('Tokenising training and validation sets...')
train_dataset = SMSDataset(train_df['text'], train_df['label_binary'], tokenizer)
val_dataset   = SMSDataset(val_df['text'],   val_df['label_binary'],   tokenizer)
print(f'Training samples: {len(train_dataset):,}  |  Validation samples: {len(val_dataset):,}')

# ── 4. Load pre-trained DistilBERT ────────────────────────────────────────────
# num_labels=2 : binary classification (ham=0, malicious=1)
# The model starts with general English language understanding from
# pre-training on Wikipedia + BookCorpus, then specialises on smishing data.
print('\nLoading DistilBERT model (distilbert-base-uncased)...')
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=2
)

# ── 5. Metrics function ───────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        'accuracy' : accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, average='binary', zero_division=0),
        'recall'   : recall_score(labels, preds, average='binary', zero_division=0),
        'f1'       : f1_score(labels, preds, average='binary', zero_division=0)
    }

# ── 6. Training configuration ─────────────────────────────────────────────────
# num_train_epochs=3    : three full passes through the training messages
# learning_rate=2e-5    : standard fine-tuning rate for BERT-family models
# warmup_steps=200      : replaces warmup_ratio (renamed in transformers>=4.36)
#                         200 steps of gradual LR warmup prevents early instability
# weight_decay=0.01     : L2 regularisation to reduce overfitting
# load_best_model_at_end=True : saves the checkpoint with the best F1, not the last
#
# GPU / CPU:
#   Google Colab with GPU : Colab detects GPU automatically -- no flag needed
#   Local CPU only        : uncomment use_cpu=True below
#
# NOTE: class imbalance (81/19) is partially handled by the model cross-entropy
# loss during fine-tuning. If recall on malicious class is below 0.70 after
# training, consider adding class_weight to the loss function.

import torch
USE_GPU = torch.cuda.is_available()
print(f'\nDevice: {"GPU (" + torch.cuda.get_device_name(0) + ")" if USE_GPU else "CPU"}')
print(f'Training will use: {"GPU -- much faster" if USE_GPU else "CPU -- expect 1-2 hours"}')

training_args = TrainingArguments(
    output_dir='models/distilbert',
    num_train_epochs=3,
    per_device_train_batch_size=16,   # reduce to 8 if you get CUDA out of memory error
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    warmup_steps=200,                 # works on all transformers versions
    weight_decay=0.01,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    logging_steps=50,
    seed=42,
    # use_cpu=True                    # uncomment ONLY if running on CPU with no GPU
)

# ── 7. Train ──────────────────────────────────────────────────────────────────
print('\nStarting DistilBERT fine-tuning...')
print('Training takes 1 to 2 hours on CPU. Progress prints every 50 steps.')
print('You should see loss decreasing and F1 increasing across the 3 epochs.\n')

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)
trainer.train()
print('\nDistilBERT fine-tuning complete.')

# ── 8. Evaluate on validation set ────────────────────────────────────────────
print('\nRunning final evaluation on validation set...')
preds_output = trainer.predict(val_dataset)
y_pred = np.argmax(preds_output.predictions, axis=-1)
y_true = val_df['label_binary'].values

acc   = accuracy_score(y_true, y_pred)
prec  = precision_score(y_true, y_pred, average='binary', zero_division=0)
rec   = recall_score(y_true, y_pred, average='binary', zero_division=0)
f1v   = f1_score(y_true, y_pred, average='binary', zero_division=0)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
fpr   = fp / (fp + tn)

print(f'\n===== DistilBERT Results =====')
print(f'Accuracy            : {acc:.4f}')
print(f'Precision           : {prec:.4f}')
print(f'Recall              : {rec:.4f}')
print(f'F1-Score            : {f1v:.4f}')
print(f'False Positive Rate : {fpr:.4f}')
print(f'\nConfusion Matrix:')
print(f'  TN={tn}  FP={fp}')
print(f'  FN={fn}  TP={tp}')
print(f'\nPer-class report:')
print(classification_report(y_true, y_pred, target_names=['Ham (0)', 'Malicious (1)']))

# ── 9. Save fine-tuned model and results ──────────────────────────────────────
os.makedirs('models/distilbert_final', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

trainer.save_model('models/distilbert_final')
tokenizer.save_pretrained('models/distilbert_final')

pd.DataFrame([{
    'model'    : 'DistilBERT',
    'accuracy' : acc,
    'precision': prec,
    'recall'   : rec,
    'f1'       : f1v,
    'fpr'      : fpr,
    'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
}]).to_csv('outputs/distilbert_results.csv', index=False)

print('\nModel saved  : models/distilbert_final/')
print('Results saved: outputs/distilbert_results.csv')

# ── 10. SHAP Computation ──────────────────────────────────────────────────────
# SHAP (SHapley Additive exPlanations) identifies which tokens in each message
# most influenced DistilBERT's prediction.
# These scores are used in your explainability evaluation to compare
# DistilBERT's token-level importance against your LLM agent's CoT reasoning.
#
# Strategy: balanced sample of 50 malicious + 50 ham = 100 messages total.
# Running SHAP on all 893 validation messages would take several hours.
# 100 is sufficient for comparison purposes -- note this in your methodology.
print('\n' + '='*60)
print('SHAP COMPUTATION')
print('='*60)
print('Computing SHAP values on a balanced 100-message sample...')
print('This takes 10 to 20 minutes. Please wait.\n')

# Build balanced sample: 50 from each class
malicious_idx = val_df[val_df['label_binary'] == 1].index[:50].tolist()
ham_idx       = val_df[val_df['label_binary'] == 0].index[:50].tolist()
shap_sample   = val_df.loc[malicious_idx + ham_idx].reset_index(drop=True)

print(f'SHAP sample: {len(malicious_idx)} malicious + {len(ham_idx)} ham '
      f'= {len(shap_sample)} messages total')

# Load the saved model as a Hugging Face pipeline for SHAP
pipe = pipeline(
    'text-classification',
    model='models/distilbert_final',
    tokenizer='models/distilbert_final',
    return_all_scores=True,
    device=-1        # -1 = CPU. Change to 0 if you have a GPU.
)

# Create SHAP explainer and compute values
# shap.Explainer automatically selects the Partition explainer for text pipelines
explainer   = shap.Explainer(pipe)
shap_values = explainer(shap_sample['text'].tolist())

# Save SHAP values for use in the explainability evaluation phase
with open('outputs/distilbert_shap_values.pkl', 'wb') as f:
    pickle.dump(shap_values, f)

# Save the sample with labels so you know which message is which
shap_sample.to_csv('outputs/shap_sample.csv', index=False)

print('\nSHAP values saved : outputs/distilbert_shap_values.pkl')
print('SHAP sample saved : outputs/shap_sample.csv')
print('\nDistilBERT script complete.')
print('\nNext step: run python src/evaluate_baselines.py')
