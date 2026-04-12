import pandas as pd
import re
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 42

INPUT = r"C:\Users\kpast\OneDrive\Dokumenty\deep\data\merged.parquet"
MIN_LEN = 200

df = pd.read_parquet(INPUT)
print(f"Loaded: {df.shape}")

cols = df.columns.tolist()

# basic filters
if "extract_success" in cols:
    df = df[df["extract_success"] == True]
if "extract_method" in cols:
    df = df[~df["extract_method"].astype(str).str.contains("download_failed", case=False, na=False)]

assert "text" in df.columns, "No 'text' column found"

df = df[df["text"].notna()].copy()

def clean_text(t):
    t = str(t).replace("\ufeff", " ").replace("ï»¿", " ")
    return re.sub(r"\s+", " ", t).strip()

df["text_clean"] = df["text"].apply(clean_text)
df = df[df["text_clean"].str.len() >= MIN_LEN]

# deduplication
dedup_col = next((c for c in ["url_norm", "url"] if c in cols), None)
if dedup_col:
    df = df.drop_duplicates(subset=[dedup_col])
df = df.drop_duplicates(subset=["text_clean"])
print(f"After cleaning + dedup: {df.shape}")

# quality metrics
def alpha_ratio(t):
    return sum(c.isalpha() for c in t) / len(t) if t else 0.0

def latin_ratio(t):
    letters = [c for c in t if c.isalpha()]
    return sum(bool(re.match(r"[A-Za-zÀ-ÿ]", c)) for c in letters) / len(letters) if letters else 0.0

def weird_ratio(t):
    return sum(c in "ï»¿" for c in t) / len(t) if t else 0.0

df["alpha_ratio"] = df["text_clean"].apply(alpha_ratio)
df["latin_ratio"] = df["text_clean"].apply(latin_ratio)
df["weird_char_ratio"] = df["text_clean"].apply(weird_ratio)
df["text_len_clean"] = df["text_clean"].str.len()

df["quality_flag"] = "ok"
df.loc[df["alpha_ratio"] < 0.50, "quality_flag"] = "low_alpha"
df.loc[df["weird_char_ratio"] > 0.01, "quality_flag"] = "encoding_noise"

# language detection
def detect_lang(t):
    try:
        return detect(t[:3000])
    except:
        return "unknown"

print(f"Running language detection on {len(df)} rows...")
df["lang"] = df["text_clean"].apply(detect_lang)
df["is_english"] = (df["lang"] == "en") & (df["latin_ratio"] >= 0.85) & (df["quality_flag"] == "ok")

print(df["lang"].value_counts().head(20))
print(df["is_english"].value_counts())

# output
keep = [c for c in [
    "news_date", "first_timestamp_utc", "last_timestamp_utc", "url", "url_norm",
    "source", "domain", "extract_success", "extract_method", "text_len", "text_len_clean",
    "alpha_ratio", "latin_ratio", "weird_char_ratio", "quality_flag", "lang", "is_english",
    "text_clean", "is_core_climate", "is_policy_regulation", "is_physical_risk",
    "is_energy_geopolitics", "is_cleantech", "is_activism_litigation",
] if c in df.columns]

df[keep].to_csv("cleaned_articles_full.csv", index=False)
df[df["is_english"]][keep].to_csv("cleaned_articles_english.csv", index=False)
df["lang"].value_counts().rename_axis("lang").reset_index(name="count").to_csv("language_summary.csv", index=False)

print("Done")