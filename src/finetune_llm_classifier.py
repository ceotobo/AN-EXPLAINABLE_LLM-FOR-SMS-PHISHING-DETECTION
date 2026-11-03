# finetune_llm_classifier.py -- LoRA fine-tuning of Mistral-7B and Llama-3-8B
# as binary classifiers for smishing detection.
#
# Method: embedding-based fine-tuning (arXiv:2512.12677)
#   - Load base LLM in 4-bit NF4 quantization
#   - Append EOS token; take final-token hidden state as sequence representation
#   - Attach a 2-layer classification head
#   - Train only the head + LoRA adapters injected into attention/FFN
#
# Run with: python src/finetune_llm_classifier.py
#
# Requires: pip install unsloth trl peft bitsandbytes transformers datasets
import unsloth
import os, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════
MODELS_TO_TRAIN = [
    ("unsloth/llama-3-8b-Instruct-bnb-4bit", "llama3"),
]

MAX_SEQ_LEN   = 256
BATCH_SIZE    = 4
GRAD_ACCUM    = 4
EPOCHS        = 3
LR            = 2e-4
LORA_R        = 16
LORA_ALPHA    = 16
NUM_LABELS    = 2


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════
class SMSClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=MAX_SEQ_LEN):
        self.texts = [str(t) for t in texts]
        self.labels = [int(l) for l in labels]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        # Append EOS token id at the end (the paper appends EOS if not present)
        input_ids = enc["input_ids"].squeeze(0)
        attn_mask = enc["attention_mask"].squeeze(0)

        # Find length of actual content (before padding)
        length = int(attn_mask.sum())
        if length < self.max_len:
            # Place EOS right after the content
            input_ids[length] = self.tokenizer.eos_token_id
            attn_mask[length] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Model with classification head on final-token embedding
# ══════════════════════════════════════════════════════════════════════════════
class LLMClassifier(nn.Module):
    """
    Wraps a causal LLM with a classification head on the final-token
    hidden state (analogous to BERT's [CLS] token).
    """
    def __init__(self, base_model, hidden_size, num_labels=NUM_LABELS,
                 dropout=0.1):
        super().__init__()
        self.base = base_model
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_labels),
        )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # Last layer hidden states: (batch, seq_len, hidden)
        hidden = outputs.hidden_states[-1]

        # Get the hidden state at the position of the last non-pad token
        # attention_mask: (batch, seq_len)
        lengths = attention_mask.sum(dim=1) - 1  # index of last real token
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        last_hidden = hidden[batch_idx, lengths]  # (batch, hidden)

        logits = self.classifier(self.dropout(last_hidden))

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {"loss": loss, "logits": logits}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        "accuracy":  accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall":    recall_score(labels, preds, zero_division=0),
        "f1":        f1_score(labels, preds, zero_division=0),
        "fpr":       fp / (fp + tn) if (fp + tn) > 0 else 0.0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Training pipeline for one model
# ══════════════════════════════════════════════════════════════════════════════
def train_one_model(model_name, short_name, train_df, val_df, test_df):
    print("\n" + "=" * 70)
    print(f"TRAINING: {short_name}  ({model_name})")
    print("=" * 70)

    from unsloth import FastLanguageModel

    # --- Load base model with LoRA ---
    print("Loading base model ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --- Attach LoRA adapters ---
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # --- Wrap with classification head ---
    hidden_size = model.config.hidden_size
    classifier = LLMClassifier(model, hidden_size)

    # Freeze base LoRA-irrelevant params; keep LoRA + classifier trainable
    trainable = 0
    total = 0
    for name, param in classifier.named_parameters():
        total += param.numel()
        if "lora_" in name or "classifier" in name:
            param.requires_grad = True
            trainable += param.numel()
        else:
            param.requires_grad = False
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # --- Datasets ---
    train_ds = SMSClassificationDataset(train_df["text"], train_df["label_binary"], tokenizer)
    val_ds   = SMSClassificationDataset(val_df["text"],   val_df["label_binary"],   tokenizer)
    test_ds  = SMSClassificationDataset(test_df["text"],  test_df["label_binary"],  tokenizer)

    # --- Training arguments ---
    out_dir = f"models/{short_name}_smishing_lora"
    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        warmup_steps=50,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        seed=42,
        report_to="none",
    )

    trainer = Trainer(
        model=classifier,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    print("Starting training ...")
    t0 = time.time()
    trainer.train()
    print(f"Training complete in {(time.time()-t0)/60:.1f} min")

    # --- Test set evaluation ---
    print("\nEvaluating on test set ...")
    test_results = trainer.evaluate(test_ds)
    print(f"  Accuracy:  {test_results['eval_accuracy']:.4f}")
    print(f"  Precision: {test_results['eval_precision']:.4f}")
    print(f"  Recall:    {test_results['eval_recall']:.4f}")
    print(f"  F1:        {test_results['eval_f1']:.4f}")
    print(f"  FPR:       {test_results['eval_fpr']:.4f}")

    # --- Save LoRA adapter + classifier head ---
    os.makedirs(out_dir, exist_ok=True)
    # Save the classifier head separately
    torch.save(classifier.classifier.state_dict(),
               os.path.join(out_dir, "classifier_head.pt"))
    # Save LoRA adapter via PEFT
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    metrics = {
        "model": short_name,
        "base": model_name,
        "accuracy":  round(test_results["eval_accuracy"], 4),
        "precision": round(test_results["eval_precision"], 4),
        "recall":    round(test_results["eval_recall"], 4),
        "f1":        round(test_results["eval_f1"], 4),
        "fpr":       round(test_results["eval_fpr"], 4),
    }

    # Free memory
    del model, classifier, trainer
    torch.cuda.empty_cache()

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("LoRA FINE-TUNING: MISTRAL-7B AND LLAMA-3-8B AS CLASSIFIERS")
    print("=" * 70)

    # --- Load data ---
    train_df = pd.read_csv("data/train.csv")
    val_df   = pd.read_csv("data/val.csv")
    test_df  = pd.read_csv("data/test.csv")

    print(f"Train: {len(train_df)}  |  Val: {len(val_df)}  |  Test: {len(test_df)}")
    print(f"Train balance: {train_df['label_binary'].value_counts().to_dict()}")
    print(f"Test balance:  {test_df['label_binary'].value_counts().to_dict()}")

    # --- Train each model sequentially ---
    all_metrics = []
    for model_name, short_name in MODELS_TO_TRAIN:
        try:
            metrics = train_one_model(model_name, short_name,
                                      train_df, val_df, test_df)
            all_metrics.append(metrics)
        except Exception as e:
            print(f"\nERROR training {short_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # --- Summary ---
    print("\n" + "=" * 70)
    print("FINAL RESULTS -- Fine-tuned LLM Classifiers")
    print("=" * 70)
    summary = pd.DataFrame(all_metrics)
    print(summary.to_string(index=False))
    summary.to_csv("outputs/finetuned_llm_results.csv", index=False)
    print("\nSaved: outputs/finetuned_llm_results.csv")


if __name__ == "__main__":
    main()