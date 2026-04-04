-- news_daily_panel_v2.sql
--
-- Final baseline BigQuery query for the climate-tail-risk project.
--
-- Purpose:
-- Build a daily climate/transition-related news panel from GDELT GKG 2.0
-- for merge with daily WTI market features.
--
-- Output:
-- One row per day with:
-- - article counts
-- - tone summaries
-- - category counts
-- - category shares
--
-- Notes:
-- - This is the baseline production query used for News Dataset V2.
-- - The main modeling unit is daily.
-- - A separate raw/article-level audit path is retained elsewhere for validation.
WITH articles AS (
  SELECT
    PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(DATE AS STRING)) AS event_timestamp,
    DATE(PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(DATE AS STRING))) AS news_date,
    DocumentIdentifier AS url,
    SourceCommonName AS source,
    LOWER(COALESCE(V2Themes, '')) AS themes,
    SAFE_CAST(SPLIT(V2Tone, ',')[OFFSET(0)] AS FLOAT64) AS tone_score

  FROM `gdelt-bq.gdeltv2.gkg_partitioned`
  WHERE _PARTITIONTIME BETWEEN TIMESTAMP('2016-03-30') AND TIMESTAMP('2024-12-31')
    AND NOT REGEXP_CONTAINS(LOWER(COALESCE(V2Themes, '')), r'business_climate')
),

tagged AS (
  SELECT
    *,

    -- Core climate / emissions / mitigation language
    IF(
      REGEXP_CONTAINS(themes, r'env_climatechange') OR
      REGEXP_CONTAINS(themes, r'wb_567_climate_change') OR
      REGEXP_CONTAINS(themes, r'ungp_climate_change_action') OR
      REGEXP_CONTAINS(themes, r'wb_579_climate_change_mitigation') OR
      REGEXP_CONTAINS(themes, r'wb_1854_methane') OR
      REGEXP_CONTAINS(themes, r'wb_1841_short_lived_climate_pollutants') OR
      REGEXP_CONTAINS(themes, r'carbon') OR
      REGEXP_CONTAINS(themes, r'emission') OR
      REGEXP_CONTAINS(themes, r'decarbon') OR
      REGEXP_CONTAINS(themes, r'net_zero'),
      1, 0
    ) AS is_climate,

    -- Carbon-specific language
    IF(
      REGEXP_CONTAINS(themes, r'carbon') OR
      REGEXP_CONTAINS(themes, r'emission') OR
      REGEXP_CONTAINS(themes, r'decarbon') OR
      REGEXP_CONTAINS(themes, r'net_zero'),
      1, 0
    ) AS is_carbon,

    -- Clean-tech / substitution themes
    IF(
      REGEXP_CONTAINS(themes, r'renewable') OR
      REGEXP_CONTAINS(themes, r'solar') OR
      REGEXP_CONTAINS(themes, r'wind') OR
      REGEXP_CONTAINS(themes, r'hydrogen') OR
      REGEXP_CONTAINS(themes, r'carboncapture') OR
      REGEXP_CONTAINS(themes, r'wb_525_renewable_energy') OR
      REGEXP_CONTAINS(themes, r'wb_528_solar_energy'),
      1, 0
    ) AS is_cleantech,

    -- Energy / fossil / oil & gas themes
    IF(
      REGEXP_CONTAINS(themes, r'energy_transition') OR
      REGEXP_CONTAINS(themes, r'fossil_fuel') OR
      REGEXP_CONTAINS(themes, r'wb_539_oil_and_gas_policy_strategy_and_institutions') OR
      REGEXP_CONTAINS(themes, r'oil') OR
      REGEXP_CONTAINS(themes, r'gas') OR
      REGEXP_CONTAINS(themes, r'energy'),
      1, 0
    ) AS is_energy,

    -- Policy / regulation / institutions
    IF(
      REGEXP_CONTAINS(themes, r'policy') OR
      REGEXP_CONTAINS(themes, r'regulation') OR
      REGEXP_CONTAINS(themes, r'carbon_tax') OR
      REGEXP_CONTAINS(themes, r'emissions_trading') OR
      REGEXP_CONTAINS(themes, r'cap_and_trade') OR
      REGEXP_CONTAINS(themes, r'wb_539_oil_and_gas_policy_strategy_and_institutions'),
      1, 0
    ) AS is_policy

  FROM articles
),

filtered AS (
  SELECT
    *,
    
    -- Final relevance rule for baseline dataset
    IF(
      is_climate = 1 OR
      (is_carbon = 1 AND is_energy = 1) OR
      (is_climate = 1 AND is_cleantech = 1) OR
      (is_climate = 1 AND is_policy = 1),
      1, 0
    ) AS is_relevant,

    -- Tone components
    IF(tone_score > 0, tone_score, NULL) AS positive_tone_component,
    IF(tone_score < 0, ABS(tone_score), NULL) AS negative_tone_component,

    -- Joint climate-energy transition flag
    IF(is_climate = 1 AND is_energy = 1, 1, 0) AS is_transition

  FROM tagged
),

daily AS (
  SELECT
    news_date,
    COUNT(*) AS n_articles,

    AVG(tone_score) AS avg_tone,
    AVG(positive_tone_component) AS avg_positivity,
    AVG(negative_tone_component) AS avg_negativity,

    SUM(is_climate) AS n_climate,
    SUM(is_carbon) AS n_carbon,
    SUM(is_cleantech) AS n_cleantech,
    SUM(is_energy) AS n_energy,
    SUM(is_policy) AS n_policy,
    SUM(is_transition) AS n_transition,

    SAFE_DIVIDE(SUM(is_climate), COUNT(*)) AS share_climate,
    SAFE_DIVIDE(SUM(is_carbon), COUNT(*)) AS share_carbon,
    SAFE_DIVIDE(SUM(is_cleantech), COUNT(*)) AS share_cleantech,
    SAFE_DIVIDE(SUM(is_energy), COUNT(*)) AS share_energy,
    SAFE_DIVIDE(SUM(is_policy), COUNT(*)) AS share_policy,
    SAFE_DIVIDE(SUM(is_transition), COUNT(*)) AS share_transition

  FROM filtered
  WHERE is_relevant = 1
  GROUP BY news_date
)

SELECT
  news_date,
  n_articles,
  avg_tone,
  avg_positivity,
  avg_negativity,
  n_climate,
  n_carbon,
  n_cleantech,
  n_energy,
  n_policy,
  n_transition,
  share_climate,
  share_carbon,
  share_cleantech,
  share_energy,
  share_policy,
  share_transition
FROM daily
ORDER BY news_date;
