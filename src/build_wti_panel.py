from pathlib import Path
import numpy as np
import pandas as pd

def build_wti_panel_from_excel(
    excel_path: Path,
    start_date: str = "2016-03-30",
    end_date: str = "2024-12-31",
    train_end_date: str = "2021-12-31",
    tail_quantile: float = 0.10,
) -> pd.DataFrame:
    # 1) Read Excel
    df = pd.read_excel(excel_path, sheet_name="Daily")

    # 2) Keep and rename columns
    df = df[["observation_date", "DCOILWTICO"]].copy()
    df = df.rename(columns={
        "observation_date": "date",
        "DCOILWTICO": "wti_price"
    })

    # 3) Clean types
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["wti_price"] = pd.to_numeric(df["wti_price"], errors="coerce")

    # 4) Drop missing and keep range
    df = df.dropna(subset=["date", "wti_price"]).copy()
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()
    df = df.sort_values("date").reset_index(drop=True)

    # 5) Build returns
    df["price_valid_for_log"] = df["wti_price"] > 0
    df["log_price"] = np.where(df["price_valid_for_log"], np.log(df["wti_price"]), np.nan)
    df["log_return"] = df["log_price"].diff()

    # 6) Lagged market features
    df["ret_lag1"] = df["log_return"].shift(1)
    df["abs_ret_lag1"] = df["log_return"].abs().shift(1)

    # 7) Rolling volatility (only past info)
    df["rv_5"] = df["log_return"].rolling(5).std().shift(1)
    df["rv_10"] = df["log_return"].rolling(10).std().shift(1)
    df["rv_22"] = df["log_return"].rolling(22).std().shift(1)

    # 8) Short-run return averages
    df["ret_mean_5"] = df["log_return"].rolling(5).mean().shift(1)
    df["ret_mean_10"] = df["log_return"].rolling(10).mean().shift(1)

    # 9) Next-day return
    df["ret_t_plus_1"] = df["log_return"].shift(-1)

    # 10) Tail-event threshold based only on training period
    train_mask = df["date"] <= pd.to_datetime(train_end_date)
    tail_threshold = df.loc[train_mask, "ret_t_plus_1"].quantile(tail_quantile)

    # 11) Binary target
    df["target_tail_t_plus_1"] = (df["ret_t_plus_1"] < tail_threshold).astype("Int64")
    df["tail_threshold_train"] = tail_threshold

    # 12) Merge key
    df["day"] = df["date"].dt.strftime("%Y-%m-%d")

    # 13) Usable rows for modelling
    predictor_cols = [
        "ret_lag1", "abs_ret_lag1", "rv_5", "rv_10", "rv_22",
        "ret_mean_5", "ret_mean_10", "ret_t_plus_1"
    ]
    df["usable_for_model"] = df[predictor_cols].notna().all(axis=1)

    # 14) Final columns
    df = df[
    [
        "day",
        "date",
        "wti_price",
        "price_valid_for_log",
        "log_price",
        "log_return",
        "ret_lag1",
        "abs_ret_lag1",
        "rv_5",
        "rv_10",
        "rv_22",
        "ret_mean_5",
        "ret_mean_10",
        "ret_t_plus_1",
        "target_tail_t_plus_1",
        "tail_threshold_train",
        "usable_for_model",
    ]
            ].copy()

    return df


def save_wti_panel(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)