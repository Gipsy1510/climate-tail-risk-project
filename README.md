# Climate Tail Risk Project

This project studies whether **climate-related news helps predict next-day downside tail risk in WTI crude oil prices**, beyond standard market-based predictors.

The workflow combines two main components:

1. a **WTI market panel** built from daily oil prices,
2. a **news pipeline** that scrapes, cleans, classifies, and aggregates climate-related news into daily signals.

The broader goal is to test whether text-based climate signals contain **incremental information** about downside risk that is not already captured by recent returns and volatility.

---

## Research Question

Do climate-related news articles improve the **next-day prediction of downside tail events in WTI oil prices** beyond market-based predictors?

---

## Project Structure

~~~text
climate-tail-risk-project/
├── data/
│   ├── raw/
│   │   ├── DCOILWTICO.xlsx
│   │   ├── complete_sample.csv
│   │   └── news_data.csv
│   └── processed/
│       ├── auto_labeled_articles.csv
│       ├── auto_labeled_full.csv
│       ├── climatebert_auto_labeled_articles.csv
│       ├── climatebert_daily_signal.csv
│       ├── daily_climate_signal.csv
│       ├── news_article_text_p1_2015_2017.parquet
│       ├── news_article_text_p2_2018_2020.parquet
│       ├── news_article_text_p3_2021_2023.parquet
│       ├── news_article_text_p4_2024_2025.parquet
│       └── wti_daily_panel.parquet
├── notebooks/
│   ├── 01_Check_wti_panel.py
│   ├── ClimateBERT.ipynb
│   └── visuals_wti.ipynb
├── sql/
├── src/
│   ├── build_wti_panel.py
│   ├── merge.py
│   ├── paths.py
│   ├── scrape_article_text_by_year.py
│   ├── cleaning/
│   │   ├── clean_and_inspect.py
│   │   └── second_clean.py
│   └── models/
│       ├── baseline.py
│       ├── baseline_model.py
│       ├── lstm.py
│       ├── merge_bert.py
│       └── train_bert.py
└── .gitignore
~~~

---

## Methodology Overview

### 1. WTI Market Panel

The market side of the project uses the FRED daily WTI spot price series to build a panel over **2015–2025**.

From the price series, the project constructs:

- daily log returns,
- lagged return,
- lagged absolute return,
- rolling realized volatility over 5, 10, and 22 days,
- short-run rolling mean returns.

The prediction target is a **next-day downside tail event**, defined as a next-day return below the **10th percentile of the training-sample next-day return distribution**.

A key implementation detail is that log prices and log returns are computed only when prices are **strictly positive**. This means that the negative WTI price episode in April 2020 generates missing values in the log-based variables and rolling volatility features. These gaps are intentional and are preserved rather than interpolated.

---

### 2. News Scraping and Text Extraction

The news side starts from a master sample of URLs and metadata. Article body text is scraped in year-based blocks using a multi-worker pipeline.

The scraper:

- reads the master URL sample,
- downloads article HTML,
- extracts body text,
- stores article-level outputs with metadata such as date, source, domain, extraction status, extraction method, tone, and text length.

The project treats this scraping workflow as the starting point for the final text pipeline rather than older exploratory source-check pipelines.

---

### 3. Cleaning Pipeline

The raw article-level outputs go through two cleaning stages.

#### First pass

The first cleaning stage keeps only usable extractions and removes:

- failed downloads,
- missing text,
- very short or low-information content,
- duplicate URLs,
- duplicate normalized URLs,
- duplicate cleaned texts.

It also computes text-quality diagnostics such as character composition ratios and formatting quality checks.

#### Second pass

The second cleaning stage focuses on higher-quality article filtering, including:

- repair of encoding artifacts,
- removal of navigation-like or archive-like pages,
- exclusion of bad domains,
- text-length restrictions,
- removal of obvious junk or placeholder patterns,
- another deduplication pass.

---

### 4. Climate Relevance Classification

Each cleaned article is assigned a climate relevance score using supervised text classification.

Two alternative daily signal constructions are implemented.

#### Baseline model

A **TF-IDF + Logistic Regression** classifier predicts whether an article is climate-related and produces article-level climate relevance probabilities.

#### ClimateBERT model

A **ClimateBERT-based classifier** provides a more semantic alternative for climate relevance prediction at the article level.

Both approaches aggregate article-level predictions into daily signals.

---

### 5. Daily Climate Signal Construction

Article-level predictions are aggregated by day to produce daily news-based predictors such as:

- total number of articles,
- number of predicted climate-related articles,
- climate intensity measures,
- log-transformed count and intensity variables,
- daily average tone,
- daily thematic averages.

Two processed daily outputs are currently available:

- `data/processed/daily_climate_signal.csv`
- `data/processed/climatebert_daily_signal.csv`

---

## Main Scripts

### `src/build_wti_panel.py`

Builds the daily WTI market panel and saves:

- `data/processed/wti_daily_panel.parquet`

### `src/scrape_article_text_by_year.py`

Scrapes article text by year block and saves article-level parquet outputs.

### `src/merge.py`

Merges the worker-level article shards into a combined article-level dataset.

### `src/cleaning/clean_and_inspect.py`

Performs the first cleaning pass on the merged article dataset.

### `src/cleaning/second_clean.py`

Performs the second cleaning pass focused on article quality and junk filtering.

### `src/models/baseline.py`

Builds the baseline article-level climate classifier and daily climate signal.

### `src/models/train_bert.py`

Builds the ClimateBERT-based article-level classifier and daily climate signal.

### Additional experimental scripts

The repository also contains additional model-side scripts such as:

- `src/models/baseline_model.py`
- `src/models/merge_bert.py`
- `src/models/lstm.py`

These complement the main pipeline and reflect additional modeling work explored in the project.

---

## Core Features

### Market features

The WTI panel includes variables such as:

- `ret_lag1`
- `abs_ret_lag1`
- `rv_5`
- `rv_10`
- `rv_22`
- `ret_mean_5`
- `ret_mean_10`
- `ret_t_plus_1`
- `target_tail_t_plus_1`

### News features

The daily news panel includes variables such as:

- `n_articles`
- `n_pred_climate`
- `climate_intensity_sum`
- `climate_intensity_mean`
- log-transformed intensity and count measures,
- tone-based features,
- thematic indicators.

A methodological caution is important here: **climate share** can be informative descriptively, but it is not treated as the main feature because it is sensitive to low daily article counts and to the composition of the sampled article universe. Raw counts and intensity measures are therefore more defensible as primary predictive inputs.

---

## Existing Processed Outputs

The repository already contains several processed outputs that make it possible to work on diagnostics, visualization, and modeling without rebuilding the full pipeline every time.

These include:

- a daily WTI market panel,
- a baseline daily climate signal,
- a ClimateBERT daily climate signal,
- article-level scraped text shards,
- auto-labeled intermediate datasets.

---

## Descriptive Analysis

The notebooks in the repository support descriptive checks and visualization, including:

- WTI spot price over time,
- daily log returns,
- realized volatility,
- tail-event timing,
- comparisons between volatility and aggregated climate-news intensity.

These figures are intended as **descriptive diagnostics**, not as proof of predictive value by themselves.

---

## Data Sources

The project uses:

- **WTI daily spot prices** from FRED,
- a **master news URL sample** with article metadata,
- scraped article body text extracted from online news sources,
- cleaned article-level datasets used for climate relevance classification.

---

## Current Status

The cleaned pipeline already includes:

- market panel construction,
- news scraping,
- article cleaning,
- article-level climate classification,
- daily climate signal construction.

The remaining empirical objective is to use these market and news panels together in a clear predictive modeling stage to evaluate whether climate-news variables improve out-of-sample prediction of next-day downside tail events.

---

## Notes

- The negative WTI price episode in April 2020 is intentionally preserved as a break in log-based features.
- Large intermediate datasets are included selectively; some heavy local-only working files may remain outside the repository.
- The repository currently prioritizes reproducible pipeline components and processed daily outputs over full-scale storage of every intermediate local dataset.

---

## Authors

- Rodrigue Mieuzet
- Melany Gipsy Moreno
- Nam Khanh Nguyen
- Katarzyna Pastuszka
