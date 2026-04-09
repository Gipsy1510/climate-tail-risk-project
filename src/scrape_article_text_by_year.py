from __future__ import annotations

import argparse
import time
from pathlib import Path
from urllib.parse import urlparse, urlsplit, urlunsplit

import pandas as pd
import requests
import trafilatura
from readability import Document


# =========================================================
# DEFAULT CONFIG
# =========================================================
DEFAULT_INPUT = Path(
    r"C:\Users\mgmorenogonz\OneDrive - Université Paris 1 Panthéon-Sorbonne\FTD\deeplearning\climate_tail_risk_project\data\raw\complete_sample.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    r"C:\Users\mgmorenogonz\OneDrive - Université Paris 1 Panthéon-Sorbonne\FTD\deeplearning\climate_tail_risk_project\data\processed\scraped_shards"
)

REQUEST_TIMEOUT = 20
MIN_TEXT_LEN = 300
CHECKPOINT_EVERY = 25
PRINT_EVERY = 10

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ClimateTailRiskProject/1.0)"
}


# =========================================================
# HELPERS
# =========================================================
def normalize_url(url: str):
    if pd.isna(url):
        return pd.NA
    url = str(url).strip()
    if not url:
        return pd.NA
    try:
        parts = urlsplit(url)
        host = parts.netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        return urlunsplit((
            parts.scheme.lower(),
            host,
            parts.path.rstrip("/"),
            "",
            ""
        ))
    except Exception:
        return url.lower()


def get_domain(url: str) -> str | None:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return None


def clean_text(text: str | None) -> str | None:
    if text is None:
        return None
    text = " ".join(str(text).split())
    return text if text else None


def download_html(url: str, session: requests.Session) -> str | None:
    try:
        r = session.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        r.raise_for_status()
        return r.text
    except Exception:
        return None


def extract_with_trafilatura_from_html(html: str, url: str) -> str | None:
    try:
        text = trafilatura.extract(
            html,
            url=url,
            favor_precision=True,
            include_comments=False,
            include_tables=False,
            include_images=False,
            output_format="txt",
        )
        return clean_text(text)
    except Exception:
        return None


def extract_with_readability_from_html(html: str, url: str) -> str | None:
    try:
        doc = Document(html)
        main_html = doc.summary()
        text = trafilatura.extract(
            main_html,
            url=url,
            favor_precision=True,
            include_comments=False,
            include_tables=False,
            include_images=False,
            output_format="txt",
        )
        return clean_text(text)
    except Exception:
        return None


def extract_article_text(url: str, session: requests.Session) -> tuple[str | None, str, bool, int]:
    html = download_html(url, session)
    if html is None:
        return (None, "download_failed", False, 0)

    text = extract_with_trafilatura_from_html(html, url)
    method = "trafilatura_html"

    if not text or len(text) < MIN_TEXT_LEN:
        fallback = extract_with_readability_from_html(html, url)
        if fallback and len(fallback) >= MIN_TEXT_LEN:
            text = fallback
            method = "readability_fallback"

    success = text is not None and len(text) >= MIN_TEXT_LEN
    text_len = len(text) if success else 0
    return (text if success else None, method, success, text_len)


def load_and_filter_input(path: Path, start_year: int, end_year: int) -> pd.DataFrame:
    df = pd.read_csv(path)

    required_cols = [
        "year",
        "news_date",
        "event_timestamp",
        "url",
        "source",
        "tone_score",
        "is_core_climate",
        "is_policy_regulation",
        "is_physical_risk",
        "is_energy_geopolitics",
        "is_cleantech",
        "is_activism_litigation",
        "n_active_buckets",
        "priority_score",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[(df["year"] >= start_year) & (df["year"] <= end_year)].copy()
    if df.empty:
        raise ValueError(f"No rows found for years {start_year}-{end_year}")

    df = df.dropna(subset=["url"]).copy()
    df["url"] = df["url"].astype(str).str.strip()

    if "url_norm" not in df.columns:
        df["url_norm"] = df["url"].apply(normalize_url)

    df = df.drop_duplicates(subset=["news_date", "url_norm"]).copy()
    df["domain"] = df["url"].apply(get_domain)

    return df.reset_index(drop=True)


def maybe_resume(df: pd.DataFrame, checkpoint_path: Path) -> tuple[pd.DataFrame, list[dict]]:
    if not checkpoint_path.exists():
        return df, []

    old = pd.read_parquet(checkpoint_path)
    done_keys = set(old["news_date"].astype(str) + "||" + old["url_norm"].astype(str))

    current_keys = df["news_date"].astype(str) + "||" + df["url_norm"].astype(str)
    remaining = df.loc[~current_keys.isin(done_keys)].copy().reset_index(drop=True)
    existing_rows = old.to_dict(orient="records")

    print(f"Checkpoint found: {len(old):,} rows already processed")
    print(f"Remaining rows: {len(remaining):,}")

    return remaining, existing_rows


def save_checkpoint(rows: list[dict], checkpoint_path: Path) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(checkpoint_path, index=False)


# =========================================================
# MAIN
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--worker", type=str, default="worker")
    parser.add_argument("--input-path", type=str, default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_name = f"{args.worker}_{args.start_year}_{args.end_year}"
    output_path = output_dir / f"news_article_text_{shard_name}.parquet"
    checkpoint_path = output_dir / f"news_article_text_{shard_name}_checkpoint.parquet"

    df = load_and_filter_input(input_path, args.start_year, args.end_year)
    print(f"Loaded {len(df):,} rows for years {args.start_year}-{args.end_year}")

    df, rows = maybe_resume(df, checkpoint_path)

    session = requests.Session()
    total = len(df)
    start_time = time.time()

    for i, row in df.iterrows():
        text, method, success, text_len = extract_article_text(row["url"], session)

        rows.append(
            {
                "year": row["year"],
                "news_date": row["news_date"],
                "event_timestamp": row["event_timestamp"],
                "url": row["url"],
                "url_norm": row["url_norm"],
                "domain": row["domain"],
                "source": row["source"],
                "tone_score": row["tone_score"],
                "is_core_climate": row["is_core_climate"],
                "is_policy_regulation": row["is_policy_regulation"],
                "is_physical_risk": row["is_physical_risk"],
                "is_energy_geopolitics": row["is_energy_geopolitics"],
                "is_cleantech": row["is_cleantech"],
                "is_activism_litigation": row["is_activism_litigation"],
                "n_active_buckets": row["n_active_buckets"],
                "priority_score": row["priority_score"],
                "extract_success": success,
                "extract_method": method,
                "text_len": text_len,
                "text": text,
            }
        )

        if (i + 1) % PRINT_EVERY == 0 or (i + 1) == total:
            tmp = pd.DataFrame(rows)
            success_rate = tmp["extract_success"].mean() if len(tmp) > 0 else 0.0
            elapsed_min = (time.time() - start_time) / 60
            print(
                f"{i+1}/{total} processed | "
                f"success_rate={success_rate:.3f} | "
                f"last_method={method} | "
                f"last_text_len={text_len} | "
                f"elapsed={elapsed_min:.1f} min"
            )

        if (i + 1) % CHECKPOINT_EVERY == 0 or (i + 1) == total:
            save_checkpoint(rows, checkpoint_path)

        time.sleep(args.sleep_seconds)

    out = pd.DataFrame(rows)
    out.to_parquet(output_path, index=False)

    print("\nDone.")
    print(f"Saved final shard to: {output_path}")
    print(f"Saved checkpoint to: {checkpoint_path}")
    print(f"Final success rate: {out['extract_success'].mean():.3f}")
    print(out["text_len"].describe())


if __name__ == "__main__":
    main()
