from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import urlretrieve

import numpy as np
import pandas as pd


# =========================================================
# AUDIT / BACKUP PIPELINE CONFIG
# =========================================================
# This module is retained for article-level validation and
# backup construction from raw GDELT daily ZIP files.
#
# It is NOT the main baseline production route anymore.
# The main baseline news pipeline uses BigQuery daily
# aggregation to build the final merge-ready daily panel.
#
# This module is useful for:
# 1) validating raw GDELT bulk structure,
# 2) checking article-level relevance flags on short windows,
# 3) producing backup daily features from downloaded ZIP files.
# =========================================================

# GKG 1.0 daily bulk format (11 columns, header row)
GKG1_COLNAMES = [
    "DATE",
    "NUMARTS",
    "COUNTS",
    "THEMES",
    "LOCATIONS",
    "PERSONS",
    "ORGANIZATIONS",
    "TONE",
    "CAMEOEVENTIDS",
    "SOURCES",
    "SOURCEURLS",
]

# Columns needed for the article-level audit route
USECOLS = ["DATE", "THEMES", "TONE", "SOURCES", "SOURCEURLS"]

MAX_WORKERS = 4


# =========================================================
# DATE HELPERS
# =========================================================
def iter_days(start_day: str, end_day: str):
    start = datetime.strptime(start_day, "%Y-%m-%d")
    end = datetime.strptime(end_day, "%Y-%m-%d")
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


# =========================================================
# DOWNLOAD HELPERS
# =========================================================
def _download_daily_gkg(day_str: str, download_dir: Path):
    stamp = day_str.replace("-", "")
    url = f"http://data.gdeltproject.org/gkg/{stamp}.gkg.csv.zip"
    dest = download_dir / f"{stamp}.gkg.csv.zip"

    if dest.exists():
        return ("already_exists", dest)

    try:
        urlretrieve(url, dest)
        return ("downloaded", dest)
    except HTTPError as e:
        return ("missing" if e.code == 404 else "error", str(e))
    except URLError as e:
        return ("error", str(e))


def download_gdelt_day(day_str: str, download_dir: Path, max_workers: int = 1):
    download_dir.mkdir(parents=True, exist_ok=True)
    status, info = _download_daily_gkg(day_str, download_dir)
    print(f"  [{status}] {info}")
    return {status: 1}


def get_day_zip_files(day_str: str, download_dir: Path):
    stamp = day_str.replace("-", "")
    path = download_dir / f"{stamp}.gkg.csv.zip"
    return [path] if path.exists() else []


# =========================================================
# CLEANING HELPERS
# =========================================================
def normalize_url(url):
    if pd.isna(url):
        return pd.NA

    url = str(url).strip()
    if not url:
        return pd.NA

    try:
        parts = urlsplit(url)
        normalized = urlunsplit(
            (
                parts.scheme.lower(),
                parts.netloc.lower(),
                parts.path.rstrip("/"),
                "",
                "",
            )
        )
        return normalized
    except Exception:
        return url.lower()


def shannon_entropy(counts):
    counts = np.array(counts, dtype=float)
    counts = counts[counts > 0]
    if len(counts) == 0:
        return 0.0
    probs = counts / counts.sum()
    return float(-(probs * np.log(probs)).sum())


# =========================================================
# READ ONE ZIP
# =========================================================
def read_gkg_zip(zip_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        zip_path,
        sep="\t",
        header=0,  # GKG 1.0 daily bulk has a real header row
        usecols=USECOLS,
        dtype="string",
        on_bad_lines="skip",
    )

    # Normalize names
    df.columns = [c.lower() for c in df.columns]

    # Align naming with rest of the pipeline
    df = df.rename(
        columns={
            "date": "date",
            "themes": "v1_themes",
            "tone": "tone_raw",
            "sources": "source_common_name",
            "sourceurls": "document_identifier",
        }
    )
    return df


# =========================================================
# CLEAN ONE RAW FRAME
# =========================================================
def clean_gkg_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # GKG 1.0 daily bulk has day-level date only
    df["date"] = df["date"].astype("string")
    df["usable_timestamp"] = df["date"]
    df["usable_timestamp_dt"] = pd.to_datetime(
        df["date"], format="%Y%m%d", errors="coerce"
    )

    # Fields kept for consistency with the broader project schema
    df["has_precise_timestamp"] = False
    df["timestamp_mismatch"] = False
    df["page_precise_pubtimestamp"] = pd.NA
    df["extras_xml"] = pd.NA

    # Parse tone string: overall,pos,neg,polarity,actref,selfref
    tone_parts = df["tone_raw"].str.split(",", expand=True)
    df["tone_overall"] = pd.to_numeric(tone_parts[0], errors="coerce")
    df["tone_positive"] = pd.to_numeric(tone_parts[1], errors="coerce")
    df["tone_negative"] = pd.to_numeric(tone_parts[2], errors="coerce")
    df["tone_polarity"] = pd.to_numeric(tone_parts[3], errors="coerce")

    # URL normalization and dedup key
    df["normalized_url"] = df["document_identifier"].apply(normalize_url)
    df["dedup_key"] = df["normalized_url"].fillna(
        df["date"].astype("string") + "_" + df.index.astype("string")
    )

    df["v1_themes"] = df["v1_themes"].fillna("")

    # Synthetic record id because GKG 1.0 has no article record id
    df["gkg_record_id"] = df["date"].astype("string") + "_" + df.index.astype("string")

    return df


# =========================================================
# DEDUPLICATION
# =========================================================
def deduplicate_articles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep one article per normalized URL per day.
    When duplicates exist, keep the earliest usable timestamp.
    """
    df = df.copy()

    df = df.sort_values(
        by=["dedup_key", "usable_timestamp_dt", "gkg_record_id"],
        ascending=[True, True, True],
    )

    df = df.drop_duplicates(subset="dedup_key", keep="first").copy()
    return df


# =========================================================
# FIRST-PASS ARTICLE RELEVANCE FLAGS
# =========================================================
def add_relevance_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    themes = df["v1_themes"].fillna("").str.upper()

    # Transition policy / explicit climate-energy policy
    df["transition_policy"] = themes.str.contains(
        r"ENV_CLIMATECHANGE|UNGP_CLIMATE_CHANGE_ACTION|WB_567_CLIMATE_CHANGE|"
        r"ENV_CARBON|ENV_GREENHOUSE|ENV_ENERGYPOLICY|"
        r"PARIS_AGREEMENT|NET_ZERO|CARBON_TAX|"
        r"EPU_POLICY_ENERGY|EPU_POLICY_ENVIRONMENT",
        regex=True,
        na=False,
    )

    # Physical climate risk (not broad geological hazards)
    df["physical_risk"] = themes.str.contains(
        r"NATURAL_DISASTER_FLOOD|NATURAL_DISASTER_FLOODED|"
        r"NATURAL_DISASTER_DROUGHT|NATURAL_DISASTER_WILDFIRE|"
        r"NATURAL_DISASTER_WILDFIRES|NATURAL_DISASTER_BUSHFIRE|"
        r"NATURAL_DISASTER_BUSHFIRES|NATURAL_DISASTER_CYCLONE|"
        r"NATURAL_DISASTER_HURRICANE|NATURAL_DISASTER_TORNADO|"
        r"NATURAL_DISASTER_TORNADOES|NATURAL_DISASTER_HEATWAVE|"
        r"NATURAL_DISASTER_SEVERE_WEATHER|NATURAL_DISASTER_HIGH_WINDS|"
        r"ENV_FLOOD|ENV_DROUGHT|ENV_WILDFIRE|ENV_STORM|"
        r"ENV_HURRICANE|ENV_HEATWAVE|ENV_EXTREMEWEATHER|"
        r"WEATHER_DROUGHT|WEATHER_FLOOD|WEATHER_WILDFIRE",
        regex=True,
        na=False,
    )

    # Energy supply / extractives / geopolitical energy themes
    df["energy_supply_geopolitics"] = themes.str.contains(
        r"FUELPRICES|ECON_GASOLINEPRICE|ECON_OILPRICE|"
        r"WB_670_OIL|WB_671_GAS|WB_672_COAL|"
        r"WB_673_MINING|WB_699_EXTRACTIVE_INDUSTRIES|"
        r"OPEC|CRUDE_OIL|OIL_SUPPLY|NATURAL_GAS_SUPPLY|"
        r"ENV_OILSPILL|PIPELINE|LNG|EPU_POLICY_ENERGY",
        regex=True,
        na=False,
    )

    # Clean tech / renewables / substitution
    df["clean_tech"] = themes.str.contains(
        r"ENV_SOLAR|ENV_WINDPOWER|ENV_WIND|WB_525_RENEWABLE_ENERGY|"
        r"ENV_RENEWABLE|ENV_CLEANENERGY|ENV_HYDROGEN|"
        r"ENV_GREEN|WB_RENEWABLE|ELECTRIC_VEHICLE|ENV_BATTERY",
        regex=True,
        na=False,
    )

    # Activism / litigation / divestment
    df["activism_litigation"] = themes.str.contains(
        r"ENV_ACTIVISM|CLIMATE_LITIGATION|DIVESTMENT|"
        r"ENV_PROTEST|FRIDAYS_FOR_FUTURE|"
        r"UNGP_BUSINESS_HUMAN_RIGHTS",
        regex=True,
        na=False,
    )

    channel_cols = [
        "transition_policy",
        "physical_risk",
        "energy_supply_geopolitics",
        "clean_tech",
        "activism_litigation",
    ]

    df["candidate_relevant"] = df[channel_cols].any(axis=1)
    df["candidate_relevant_strict"] = df["candidate_relevant"]
    df["n_active_channels"] = df[channel_cols].sum(axis=1)

    df["n_themes"] = (
        df["v1_themes"]
        .str.strip(";")
        .str.split(";")
        .apply(lambda x: len([v for v in x if v]) if isinstance(x, list) else 0)
    )

    return df


# =========================================================
# ARTICLE-LEVEL OUTPUT
# =========================================================
def build_article_level_day(df: pd.DataFrame, day_str: str) -> pd.DataFrame:
    df = df.copy()

    keep_cols = [
        "gkg_record_id",
        "date",
        "page_precise_pubtimestamp",
        "usable_timestamp",
        "usable_timestamp_dt",
        "has_precise_timestamp",
        "timestamp_mismatch",
        "source_common_name",
        "document_identifier",
        "normalized_url",
        "dedup_key",
        "v1_themes",
        "n_themes",
        "candidate_relevant",
        "candidate_relevant_strict",
        "n_active_channels",
        "transition_policy",
        "physical_risk",
        "energy_supply_geopolitics",
        "clean_tech",
        "activism_litigation",
        "tone_overall",
        "tone_positive",
        "tone_negative",
        "tone_polarity",
    ]

    out = df[keep_cols].copy()
    out["day"] = day_str
    return out


# =========================================================
# DAILY FEATURE AGGREGATION
# =========================================================
def aggregate_daily_features(
    raw_df: pd.DataFrame,
    dedup_df: pd.DataFrame,
    article_df: pd.DataFrame,
    day_str: str,
) -> pd.DataFrame:
    raw_n = int(len(raw_df))
    dedup_n = int(len(dedup_df))
    relevant_n = int(article_df["candidate_relevant"].sum())
    relevant_strict_n = int(article_df["candidate_relevant_strict"].sum())

    channel_cols = [
        "transition_policy",
        "physical_risk",
        "energy_supply_geopolitics",
        "clean_tech",
        "activism_litigation",
    ]

    channel_counts = [int(article_df[c].sum()) for c in channel_cols]
    entropy_channels = shannon_entropy(channel_counts)

    out = pd.DataFrame(
        {
            "day": [day_str],
            "n_articles_raw": [raw_n],
            "n_articles_dedup": [dedup_n],
            "n_relevant_articles": [relevant_n],
            "n_relevant_articles_strict": [relevant_strict_n],
            "share_relevant_articles": [relevant_n / dedup_n if dedup_n > 0 else 0.0],
            "share_precise_timestamp_raw": [raw_df["has_precise_timestamp"].mean()],
            "share_timestamp_mismatch_raw": [raw_df["timestamp_mismatch"].mean()],
            "transition_policy_count": [int(article_df["transition_policy"].sum())],
            "physical_risk_count": [int(article_df["physical_risk"].sum())],
            "energy_supply_geopolitics_count": [
                int(article_df["energy_supply_geopolitics"].sum())
            ],
            "clean_tech_count": [int(article_df["clean_tech"].sum())],
            "activism_litigation_count": [
                int(article_df["activism_litigation"].sum())
            ],
            "channel_entropy": [entropy_channels],
            "avg_active_channels_per_article": [
                article_df["n_active_channels"].mean() if len(article_df) else 0.0
            ],
            "avg_themes_per_article": [
                article_df["n_themes"].mean() if len(article_df) else 0.0
            ],
            "tone_mean": [
                article_df["tone_overall"].mean() if "tone_overall" in article_df else None
            ],
            "tone_std": [
                article_df["tone_overall"].std() if "tone_overall" in article_df else None
            ],
            "tone_negative_mean": [
                article_df["tone_negative"].mean() if "tone_negative" in article_df else None
            ],
        }
    )

    return out


# =========================================================
# PROCESS ONE DAY END-TO-END
# =========================================================
def process_one_day(day_str: str, raw_news_dir: Path):
    zip_files = get_day_zip_files(day_str, raw_news_dir)

    if not zip_files:
        raise FileNotFoundError(f"No GKG zip files found for {day_str} in {raw_news_dir}")

    raw_frames = []
    for zp in zip_files:
        try:
            raw_df = read_gkg_zip(zp)
            raw_df = clean_gkg_frame(raw_df)
            raw_frames.append(raw_df)
        except Exception as e:
            print(f"Problem reading {zp.name}: {e}")

    if not raw_frames:
        raise ValueError(f"All files failed for {day_str}")

    raw_day = pd.concat(raw_frames, ignore_index=True)
    raw_day["day"] = day_str

    dedup_day = deduplicate_articles(raw_day)
    dedup_day = add_relevance_flags(dedup_day)

    article_day = build_article_level_day(dedup_day, day_str)
    daily_features = aggregate_daily_features(raw_day, dedup_day, article_day, day_str)

    return raw_day, article_day, daily_features


# =========================================================
# SAVE HELPERS
# =========================================================
def save_partitioned_article_day(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    day_stamp = df["day"].iloc[0].replace("-", "")
    path = out_dir / f"article_level_{day_stamp}.parquet"
    df.to_parquet(path, index=False)
    return path


def save_daily_features(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        old = pd.read_parquet(path)
        combined = pd.concat([old, df], ignore_index=True)
        combined = combined.drop_duplicates(subset="day", keep="last").sort_values("day")
    else:
        combined = df.copy()

    combined.to_parquet(path, index=False)
    return path


# =========================================================
# FULL RANGE BUILD
# =========================================================
def build_gdelt_news_dataset(
    start_day: str,
    end_day: str,
    raw_news_dir: Path,
    processed_dir: Path,
    download_first: bool = True,
    max_workers: int = MAX_WORKERS,
    save_article_level: bool = True,
):
    """
    Article-level audit / backup pipeline for the news side.

    This function is retained for:
    1) validating raw GDELT daily bulk structure,
    2) checking article-level relevance flags on short windows,
    3) producing backup daily features from downloaded ZIP files.

    The main baseline production route for the project now uses
    BigQuery daily aggregation to build the final merge-ready
    news panel.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)

    article_dir = processed_dir / "article_level_daily"
    daily_features_path = processed_dir / "gdelt_daily_news_features.parquet"

    all_daily_features = []

    for day in iter_days(start_day, end_day):
        day_str = day.strftime("%Y-%m-%d")
        print(f"\n=== {day_str} ===")

        if download_first:
            results = download_gdelt_day(day_str, raw_news_dir, max_workers=max_workers)
            print("Download results:", results)

        try:
            raw_day, article_day, daily_features = process_one_day(day_str, raw_news_dir)

            print(
                f"raw={len(raw_day):,} | "
                f"dedup={len(article_day):,} | "
                f"relevant={int(article_day['candidate_relevant'].sum()):,}"
            )

            if save_article_level:
                save_partitioned_article_day(article_day, article_dir)

            all_daily_features.append(daily_features)

        except FileNotFoundError:
            print(f"No files found for {day_str}. Skipping.")
        except Exception as e:
            print(f"Failed on {day_str}: {e}")

    if not all_daily_features:
        raise ValueError("No daily features were created for the requested range.")

    full_daily = pd.concat(all_daily_features, ignore_index=True)
    save_daily_features(full_daily, daily_features_path)

    return full_daily