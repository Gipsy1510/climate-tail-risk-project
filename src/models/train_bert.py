# ClimateBERT ran on Google Colab because of GPU access for more efficient processing

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    DataCollatorWithPadding, Trainer, TrainingArguments,
)

LABELED_FILE = "cleaned_articles_english_v2_labeled.csv"
UNLABELED_FILE = "cleaned_articles_english.csv"
MODEL_NAME = "climatebert/distilroberta-base-climate-f"
OUTPUT_DIR = "./climatebert_finetuned"
TEXT_COL = "text_clean"
MAX_LENGTH = 512

# --- load labeled data ---
df = pd.read_csv(LABELED_FILE)
df = df[df["suggested_label"].isin(["climate_yes", "climate_no"])].copy()
df["label"] = df["suggested_label"].map({"climate_no": 0, "climate_yes": 1})
df = df[df[TEXT_COL].notna()].copy()
df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()
df = df[df[TEXT_COL].str.len() > 0]
print("Labeled shape:", df.shape)
print(df["label"].value_counts())

# --- splits ---
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df["label"])
print(f"Train: {train_df.shape} | Val: {val_df.shape} | Test: {test_df.shape}")

# --- tokenize ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch[TEXT_COL], truncation=True, max_length=MAX_LENGTH)

def to_hf(dataframe):
    ds = Dataset.from_pandas(dataframe[[TEXT_COL, "label"]], preserve_index=False)
    ds = ds.map(tokenize, batched=True).remove_columns([TEXT_COL])
    return ds

train_ds, val_ds, test_ds = to_hf(train_df), to_hf(val_df), to_hf(test_df)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# --- model ---
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2,
    id2label={0: "climate_no", 1: "climate_yes"},
    label2id={"climate_no": 0, "climate_yes": 1},
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {"accuracy": accuracy_score(labels, preds), "precision": precision, "recall": recall, "f1": f1}

# --- train ---
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch", save_strategy="epoch", logging_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="f1", greater_is_better=True,
        num_train_epochs=3, learning_rate=2e-5,
        per_device_train_batch_size=8, per_device_eval_batch_size=8,
        weight_decay=0.01, save_total_limit=2, report_to="none", fp16=False,
    ),
    train_dataset=train_ds, eval_dataset=val_ds,
    data_collator=data_collator, compute_metrics=compute_metrics,
)

trainer.train()
print("\nTest metrics:", trainer.evaluate(test_ds))

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# --- save test predictions ---
pred_out = trainer.predict(test_ds)
logits = pred_out.predictions
probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)

test_results = test_df.copy()
test_results["pred_label"] = np.argmax(logits, axis=1)
test_results["pred_prob_climate"] = probs[:, 1]
test_results["pred_name"] = test_results["pred_label"].map({0: "climate_no", 1: "climate_yes"})
test_results.to_csv("climatebert_test_predictions.csv", index=False)

# --- predict on full file ---
df2 = pd.read_csv(UNLABELED_FILE)
df2 = df2[df2[TEXT_COL].notna()].copy()
df2[TEXT_COL] = df2[TEXT_COL].astype(str).str.strip()
df2 = df2[df2[TEXT_COL].str.len() > 0]
print("Full file shape:", df2.shape)

full_ds = Dataset.from_pandas(df2[[TEXT_COL]], preserve_index=False)
full_ds = full_ds.map(tokenize, batched=True).remove_columns([TEXT_COL])

full_logits = trainer.predict(full_ds).predictions
shifted = full_logits - full_logits.max(axis=1, keepdims=True)
full_probs = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)

df2["pred_prob_climate"] = full_probs[:, 1]
df2["pred_label"] = (df2["pred_prob_climate"] >= 0.5).astype(int)
df2["pred_label_name"] = df2["pred_label"].map({0: "climate_no", 1: "climate_yes"})
df2["high_conf"] = (df2["pred_prob_climate"] >= 0.90) | (df2["pred_prob_climate"] <= 0.10)

print(df2["pred_label_name"].value_counts(normalize=True))
print(df2["high_conf"].value_counts())
df2.to_csv("climatebert_auto_labeled_articles.csv", index=False)

# --- daily signal ---
df2["news_date"] = pd.to_datetime(df2["news_date"], errors="coerce")
df2 = df2.dropna(subset=["news_date"])

daily = df2.groupby("news_date").agg(
    n_articles=(TEXT_COL, "count"),
    n_pred_climate=("pred_label", "sum"),
    climate_share=("pred_label", "mean"),
    climate_intensity_mean=("pred_prob_climate", "mean"),
    climate_intensity_sum=("pred_prob_climate", "sum"),
).reset_index()

for col in ["n_articles", "n_pred_climate", "climate_intensity_sum"]:
    daily[f"log_{col}"] = np.log1p(daily[col])

extra_cols = [c for c in ["is_core_climate", "is_policy_regulation", "is_physical_risk",
                           "is_energy_geopolitics", "is_cleantech", "is_activism_litigation",
                           "tone_score"] if c in df2.columns]
if extra_cols:
    daily = daily.merge(df2.groupby("news_date")[extra_cols].mean().reset_index(), on="news_date", how="left")

daily.to_csv("climatebert_daily_signal.csv", index=False)
print(daily.head(10).to_string(index=False))