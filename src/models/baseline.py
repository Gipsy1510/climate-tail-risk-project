import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

LABELED_FILE = "cleaned_articles_english_v2_labeled.csv"
UNLABELED_FILE = "cleaned_articles_english_v2.csv"
OUTPUT_FULL = "auto_labeled_articles.csv"
OUTPUT_HIGH_CONF = "auto_labeled_full.csv"
OUTPUT_DAILY = "daily_climate_signal.csv"

# --- load & prep labeled data ---
df = pd.read_csv(LABELED_FILE)
df = df[df["suggested_label"].isin(["climate_yes", "climate_no"])].copy()
df["label"] = df["suggested_label"].map({"climate_yes": 1, "climate_no": 0})
df = df.dropna(subset=["text_clean"]).copy()
df["text_clean"] = df["text_clean"].astype(str).str.strip()
df = df[df["text_clean"].str.len() > 0]

print("Shape:", df.shape)
print(df["label"].value_counts())
print(df["label"].value_counts(normalize=True))
print(df["text_clean"].str.len().describe())

df["news_date"] = pd.to_datetime(df["news_date"], errors="coerce")
print(df.groupby("news_date").size().describe())

# --- train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    df["text_clean"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# --- TF-IDF + logistic regression ---
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
y_prob = model.predict_proba(X_test_vec)[:, 1]

print(classification_report(y_test, y_pred))
print("Accuracy:", (y_pred == y_test).mean())
print(confusion_matrix(y_test, y_pred))
print(pd.Series(y_prob).describe())

# top features
coefs = model.coef_[0]
features = vectorizer.get_feature_names_out()
print("\nTop climate terms:")
for coef, word in sorted(zip(coefs, features), reverse=True)[:20]:
    print(f"  {word}: {coef:.3f}")
print("\nTop non-climate terms:")
for coef, word in sorted(zip(coefs, features))[:20]:
    print(f"  {word}: {coef:.3f}")

# --- predict on full unlabeled file ---
df2 = pd.read_csv(UNLABELED_FILE)
df2 = df2.dropna(subset=["text_clean"]).copy()
df2["text_clean"] = df2["text_clean"].astype(str).str.strip()
df2 = df2[df2["text_clean"].str.len() > 0]
print("Unlabeled shape:", df2.shape)

X2 = vectorizer.transform(df2["text_clean"])
df2["pred_prob_climate"] = model.predict_proba(X2)[:, 1]
df2["pred_label"] = (df2["pred_prob_climate"] >= 0.5).astype(int)
df2["pred_label_name"] = df2["pred_label"].map({1: "climate_yes", 0: "climate_no"})
df2["high_conf"] = (df2["pred_prob_climate"] >= 0.90) | (df2["pred_prob_climate"] <= 0.10)

print(df2["pred_label_name"].value_counts(normalize=True))
print(df2["high_conf"].value_counts())

df2.to_csv(OUTPUT_FULL, index=False)
df2[df2["high_conf"]].to_csv(OUTPUT_HIGH_CONF, index=False)

# --- build daily signal ---
df2["news_date"] = pd.to_datetime(df2["news_date"], errors="coerce")
df2 = df2.dropna(subset=["news_date"])

agg = {"text_clean": "count", "pred_label": ["sum", "mean"], "pred_prob_climate": ["mean", "sum"]}
daily = df2.groupby("news_date").agg(
    n_articles=("text_clean", "count"),
    n_pred_climate=("pred_label", "sum"),
    climate_share=("pred_label", "mean"),
    climate_intensity_mean=("pred_prob_climate", "mean"),
    climate_intensity_sum=("pred_prob_climate", "sum"),
    avg_tone_score=("tone_score", "mean") if "tone_score" in df2.columns else ("pred_label", "mean"),
).reset_index()

for col in ["n_articles", "n_pred_climate", "climate_intensity_sum"]:
    daily[f"log_{col}"] = np.log1p(daily[col])

cat_cols = [c for c in ["is_core_climate", "is_policy_regulation", "is_physical_risk",
                         "is_energy_geopolitics", "is_cleantech", "is_activism_litigation"]
            if c in df2.columns]
if cat_cols:
    daily = daily.merge(df2.groupby("news_date")[cat_cols].mean().reset_index(), on="news_date", how="left")

daily.to_csv(OUTPUT_DAILY, index=False)
print(daily.head(10).to_string(index=False))