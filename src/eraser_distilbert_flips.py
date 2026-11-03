# eraser_distilbert_flips.py -- binary flip-rate ERASER for DistilBERT
#
# Matches the LLM-side faithfulness test: mask the SHAP top-5 tokens,
# classify, and check whether the argmax prediction changes.
#
# Uses the same 100-message SHAP subset (outputs/shap_sample.csv) and the
# same SHAP values (outputs/distilbert_shap_values.pkl) as eraser_distilbert.py,
# but reports a binary flip rate instead of a probability drop.
#
# This flip rate is directly comparable to the LLM cited-token flip rate
# computed for config2c (0.360 mistral, 0.247 llama3).
#
# Run with: python src/eraser_distilbert_flips.py
#
# Requires:
#   models/distilbert_final/                (fine-tuned DistilBERT)
#   outputs/distilbert_shap_values.pkl      (SHAP values)
#   outputs/shap_sample.csv                 (100-message balanced subset)
#
# No Ollama, no GPU required. Runs on CPU.

import pickle, os, re, warnings
import numpy as np
import pandas as pd
import torch
from transformers import (DistilBertTokenizer,
                          DistilBertForSequenceClassification)

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading DistilBERT ...")
MODEL_PATH = os.path.abspath("models/distilbert_final")
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()
print("Model loaded.")


def predict(text):
    """Return argmax label (0 = legitimate, 1 = phishing)."""
    enc = tokenizer(text, truncation=True, padding=True,
                    max_length=128, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits
    return int(torch.argmax(logits, dim=-1).item())


def mask_tokens(text, tokens):
    """Replace each whole-word occurrence of tokens with [MASK]."""
    out = text
    for t in tokens:
        if not t.strip():
            continue
        out = re.sub(r"\b" + re.escape(t.strip()) + r"\b",
                     "[MASK]", out, flags=re.IGNORECASE)
    return out


# ── Load SHAP values and sample ──────────────────────────────────────────────
print("\nLoading SHAP values ...")
with open("outputs/distilbert_shap_values.pkl", "rb") as f:
    shap_vals = pickle.load(f)
sample_df = pd.read_csv("outputs/shap_sample.csv")
print(f"Sample: {len(sample_df)} messages "
      f"(ham={(sample_df['label_binary']==0).sum()}, "
      f"malicious={(sample_df['label_binary']==1).sum()})")


def shap_top_k(sample_idx, k=5):
    """Return the k tokens with highest absolute SHAP value for class 1."""
    tokens   = shap_vals.data[sample_idx]
    shap_v   = np.array(shap_vals.values[sample_idx])
    shap_mal = shap_v[:, 1]
    valid    = [(i, tokens[i]) for i in range(len(tokens))
                if tokens[i].strip() != ""]
    sorted_v = sorted(valid, key=lambda x: abs(shap_mal[x[0]]), reverse=True)
    return [t.strip() for _, t in sorted_v[:k]]


# ── Compute flip rates ────────────────────────────────────────────────────────
print("\nComputing flip rates ...\n")

records = []
flips = 0
attempts = 0

for i, row in sample_df.iterrows():
    sample_idx = sample_df.index.get_loc(i)
    text = str(row["text"])
    label = int(row["label_binary"])

    shap_tokens = shap_top_k(sample_idx, k=5)
    masked = mask_tokens(text, shap_tokens)

    if masked.strip() == text.strip():
        # Masking changed nothing (SHAP tokens not present in text)
        continue

    pred_orig = predict(text)
    pred_mask = predict(masked)

    flipped = (pred_orig != pred_mask)
    if flipped:
        flips += 1
    attempts += 1

    records.append({
        "idx": i,
        "label": label,
        "shap_tokens": shap_tokens,
        "pred_orig": pred_orig,
        "pred_masked": pred_mask,
        "flipped": flipped,
        "text": text[:120],
    })

    if (i + 1) % 20 == 0:
        rate = flips / attempts if attempts else 0
        print(f"  [{i+1}/{len(sample_df)}] flip rate = {rate:.3f} "
              f"({flips}/{attempts})")

df = pd.DataFrame(records)

overall  = flips / attempts if attempts else 0
phishing = df[df["label"] == 1]["flipped"].mean() if (df["label"] == 1).any() else 0
ham      = df[df["label"] == 0]["flipped"].mean() if (df["label"] == 0).any() else 0

print()
print("=" * 60)
print("ERASER FLIP RATE -- DistilBERT + SHAP top-5")
print("=" * 60)
print(f"Overall flip rate : {overall:.3f}  ({flips}/{attempts})")
print(f"Phishing flip rate: {phishing:.3f}")
print(f"Ham flip rate     : {ham:.3f}")
print()
print("Comparison:")
print(f"  SHAP + DistilBERT       : {overall:.3f}")
print(f"  LLM cited + Mistral     : 0.360")
print(f"  LLM cited + Llama3      : 0.247")

df.to_csv("outputs/eraser_distilbert_flips.csv", index=False)
print("\nSaved: outputs/eraser_distilbert_flips.csv")