from pathlib import Path
import numpy as np
import pandas as pd


def build_wti_panel_from_excel(
    excel_path: Path,
    start_date: str = "2015-01-01",
    end_date: str = "2025-12-31",
    train_end_date: str = "2022-12-31",
    tail_quantile: float = 0.10,
) -> pd.DataFrame:
    """
    Build a daily WTI panel from the FRED Excel file and create
    market-based predictors for next-day downside tail-risk prediction.
    """

    # Convert the date boundaries to pandas timestamps for filtering
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    train_end_date = pd.to_datetime(train_end_date)

    # Read the WTI daily sheet from the Excel file
    df = pd.read_excel(excel_path, sheet_name="Daily")

    # Keep the original date and WTI price columns and rename them
    df = df[["observation_date", "DCOILWTICO"]].copy()
    df = df.rename(columns={
        "observation_date": "date",
        "DCOILWTICO": "wti_price",
    })

    # Convert date and price columns to numeric/time formats
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["wti_price"] = pd.to_numeric(df["wti_price"], errors="coerce")

    # Remove rows with missing date or price, then keep the requested sample period
    df = df.dropna(subset=["date", "wti_price"]).copy()
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()

    # Sort observations chronologically
    df = df.sort_values("date").reset_index(drop=True)

    # Identify prices that can be used in log transformations
    # Log prices are only defined for strictly positive values
    df["price_valid_for_log"] = df["wti_price"] > 0

    # Compute log prices where valid; keep missing values otherwise
    df["log_price"] = np.where(
        df["price_valid_for_log"],
        np.log(df["wti_price"]),
        np.nan
    )

    # Compute daily log returns
    df["log_return"] = df["log_price"].diff()

    # Build lagged return-based predictors
    df["ret_lag1"] = df["log_return"].shift(1)
    df["abs_ret_lag1"] = df["log_return"].abs().shift(1)

    # Build rolling realized-volatility predictors using only past information
    df["rv_5"] = df["log_return"].rolling(5).std().shift(1)
    df["rv_10"] = df["log_return"].rolling(10).std().shift(1)
    df["rv_22"] = df["log_return"].rolling(22).std().shift(1)

    # Build short-run rolling mean-return predictors using only past information
    df["ret_mean_5"] = df["log_return"].rolling(5).mean().shift(1)
    df["ret_mean_10"] = df["log_return"].rolling(10).mean().shift(1)

    # Define the next-day return as the prediction target horizon
    df["ret_t_plus_1"] = df["log_return"].shift(-1)

    # Estimate the downside tail threshold using only the training period
    train_returns = df.loc[df["date"] <= train_end_date, "ret_t_plus_1"].dropna()
    tail_threshold = train_returns.quantile(tail_quantile)

    # Create the binary next-day tail-event target
    df["target_tail_t_plus_1"] = (df["ret_t_plus_1"] < tail_threshold).astype("Int64")

    # Store the training-sample threshold in the panel
    df["tail_threshold_train"] = tail_threshold

    # Create a day key for merging with daily news data
    df["day"] = df["date"].dt.strftime("%Y-%m-%d")

    # Mark rows with complete predictors and target as usable for modelling
    predictor_cols = [
        "ret_lag1",
        "abs_ret_lag1",
        "rv_5",
        "rv_10",
        "rv_22",
        "ret_mean_5",
        "ret_mean_10",
        "ret_t_plus_1",
    ]
    df["usable_for_model"] = df[predictor_cols].notna().all(axis=1)

    # Keep the final set of columns used in the WTI panel
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
    """
    Save the WTI panel as a parquet file.
    """

    # Create the output folder if it does not exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the dataframe without the index
    df.to_parquet(output_path, index=False)