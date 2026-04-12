import pandas as pd
import re

INPUT_CSV = "cleaned_articles_english.csv"
OUTPUT_CSV = "cleaned_articles_english_v2.csv"

df = pd.read_csv(INPUT_CSV)
print("Original shape:", df.shape)

def fix_encoding(text):
    if pd.isna(text):
        return text
    replacements = {
        "â€™": "'", "â€œ": '"', "â€": '"', "â€"": "-",
        "â€"": "-", "â€¦": "...", "Â ": " ", "Â": "",
        "\ufeff": " ", "ï»¿": " ",
    }
    for bad, good in replacements.items():
        text = str(text).replace(bad, good)
    return re.sub(r"\s+", " ", text).strip()

df["text_clean"] = df["text_clean"].apply(fix_encoding)

# drop navigational / non-article URLs
bad_url_patterns = r"/authors?/|/tags?/|/categor|/search|/topics?/"
bad_domains = ["upstract.com"]

bad_url = df["url"].fillna("").str.contains(bad_url_patterns, case=False, regex=True)
bad_dom = df["domain"].fillna("").isin(bad_domains)
print(f"Bad URL: {bad_url.sum()} | Bad domain: {bad_dom.sum()}")
df = df[~bad_url & ~bad_dom]

# length filter
df["text_len_v2"] = df["text_clean"].str.len()
df = df[df["text_len_v2"].between(200, 20000)]

# junk heuristics
def count_word(text, word):
    return len(re.findall(rf"\b{word}\b", str(text), re.IGNORECASE))

junk = (
    (df["text_clean"].apply(lambda x: count_word(x, "More")) > 8) |
    (df["text_clean"].apply(lambda x: count_word(x, "video")) > 8) |
    (df["text_clean"].str.contains(r"#NAME\?", case=False, na=False)) |
    (df["text_clean"].apply(lambda x: str(x).count(" - ")) > 40)
)
print(f"Junk rows removed: {junk.sum()}")
df = df[~junk]

before = len(df)
df = df.drop_duplicates(subset=["text_clean"])
print(f"Duplicates removed: {before - len(df)}")

df.drop(columns=["text_len_v2"], errors="ignore").to_csv(OUTPUT_CSV, index=False)
print("Final shape:", df.shape)
print("Saved:", OUTPUT_CSV)