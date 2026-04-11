# %%
from pathlib import Path
import sys

PROJECT_ROOT = Path.cwd().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# %%
from src.paths import RAW, PROCESSED
from src.build_wti_panel import build_wti_panel_from_excel, save_wti_panel

# %%
wti = build_wti_panel_from_excel(
    excel_path=RAW / "DCOILWTICO.xlsx",
    start_date="2015-01-01",
    end_date="2025-12-31",
    train_end_date="2022-12-31",
    tail_quantile=0.10,
)

save_wti_panel(wti, PROCESSED / "wti_daily_panel.parquet")

# %%
wti[["date", "wti_price"]].agg(["min", "max"])

# %%
bad_prices = wti.loc[wti["wti_price"] <= 0, ["date", "wti_price"]].copy()
bad_prices

# %%
wti.loc[
    (wti["date"] >= "2020-04-10") & (wti["date"] <= "2020-04-30"),
    ["date", "wti_price", "price_valid_for_log", "log_price", "log_return", "usable_for_model"]
]

# %%
wti[
    [
        "ret_lag1", "abs_ret_lag1", "rv_5", "rv_10", "rv_22",
        "ret_mean_5", "ret_mean_10", "ret_t_plus_1", "target_tail_t_plus_1"
    ]
].isna().sum()

# %%
{
    "shape": wti.shape,
    "usable_rows": int(wti["usable_for_model"].sum()),
    "start_date": wti["date"].min(),
    "end_date": wti["date"].max(),
    "tail_threshold": float(wti["tail_threshold_train"].dropna().iloc[0]),
}