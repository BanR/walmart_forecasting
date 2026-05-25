"""
Build weekly aggregate data for a TFT/N-BEATS aggregate model.

This script creates a compact weekly table from the daily M5 data. It is meant
for training an aggregate sequence model whose forecasts can be used by
src.ensemble.weekly_reconcile.

Default grouping is store-category because it is specific enough to catch
store/category bias, but much less sparse than item-store daily demand.

Output schema:

    group_id, store_id, cat_id, week_idx, week_start_d, week_end_d,
    week_start_date, week_end_date, sales, sell_price_mean, snap_days,
    event_days, christmas_days, month, year

Usage:

    python -m src.ensemble.build_weekly_aggregate \
      --group-cols store_id,cat_id \
      --output data/weekly_store_cat.parquet
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd


DATA_DIR = Path("data")
DEFAULT_OUTPUT = DATA_DIR / "weekly_store_cat.parquet"


def parse_group_cols(value: str) -> list[str]:
    cols = [c.strip() for c in value.split(",") if c.strip()]
    allowed = {"state_id", "store_id", "cat_id", "dept_id", "item_id"}
    unknown = set(cols) - allowed
    if unknown:
        raise ValueError(f"Unsupported group columns: {sorted(unknown)}")
    if not cols:
        raise ValueError("At least one group column is required")
    return cols


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sales = pd.read_csv(DATA_DIR / "sales_train_evaluation.csv")
    calendar = pd.read_csv(DATA_DIR / "calendar.csv")
    prices = pd.read_csv(DATA_DIR / "sell_prices.csv")
    return sales, calendar, prices


def build_daily_long(sales: pd.DataFrame, calendar: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Create daily long-form rows with calendar and price fields."""
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [c for c in sales.columns if c.startswith("d_")]

    daily = sales[id_cols + day_cols].melt(
        id_vars=id_cols,
        var_name="d",
        value_name="sales",
    )

    cal = calendar[
        [
            "d",
            "date",
            "wm_yr_wk",
            "month",
            "year",
            "event_name_1",
            "snap_CA",
            "snap_TX",
            "snap_WI",
        ]
    ].copy()
    cal["date"] = pd.to_datetime(cal["date"])
    cal["d_num"] = cal["d"].str.replace("d_", "", regex=False).astype(int)
    cal["week_idx"] = ((cal["d_num"] - 1) // 7).astype(int)
    cal["has_event"] = cal["event_name_1"].notna().astype(int)
    cal["is_christmas"] = ((cal["date"].dt.month == 12) & (cal["date"].dt.day == 25)).astype(int)

    daily = daily.merge(cal, on="d", how="left")

    daily["snap"] = 0
    daily.loc[daily["state_id"] == "CA", "snap"] = daily.loc[daily["state_id"] == "CA", "snap_CA"].values
    daily.loc[daily["state_id"] == "TX", "snap"] = daily.loc[daily["state_id"] == "TX", "snap_TX"].values
    daily.loc[daily["state_id"] == "WI", "snap"] = daily.loc[daily["state_id"] == "WI", "snap_WI"].values
    daily = daily.drop(columns=["snap_CA", "snap_TX", "snap_WI"])

    daily = daily.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    daily["sell_price"] = daily["sell_price"].fillna(0)

    return daily


def build_weekly_aggregate(daily: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Aggregate daily item-store rows to weekly group rows."""
    daily["revenue"] = daily["sales"] * daily["sell_price"]

    grouped = daily.groupby(group_cols + ["week_idx"], as_index=False).agg(
        sales=("sales", "sum"),
        revenue=("revenue", "sum"),
        sell_price_mean=("sell_price", "mean"),
        snap_days=("snap", "sum"),
        event_days=("has_event", "sum"),
        christmas_days=("is_christmas", "sum"),
        week_start_d=("d_num", "min"),
        week_end_d=("d_num", "max"),
        week_start_date=("date", "min"),
        week_end_date=("date", "max"),
        month=("month", "first"),
        year=("year", "first"),
    )

    grouped["group_id"] = grouped[group_cols].astype(str).agg("_".join, axis=1)
    ordered_cols = [
        "group_id",
        *group_cols,
        "week_idx",
        "week_start_d",
        "week_end_d",
        "week_start_date",
        "week_end_date",
        "sales",
        "revenue",
        "sell_price_mean",
        "snap_days",
        "event_days",
        "christmas_days",
        "month",
        "year",
    ]
    return grouped[ordered_cols].sort_values(["group_id", "week_idx"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-cols", default="store_id,cat_id")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    t0 = time.time()
    group_cols = parse_group_cols(args.group_cols)
    print(f"Building weekly aggregate for group_cols={group_cols}")

    sales, calendar, prices = load_inputs()
    daily = build_daily_long(sales, calendar, prices)
    weekly = build_weekly_aggregate(daily, group_cols)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    weekly.to_parquet(args.output, index=False)

    print(f"Saved: {args.output}")
    print(f"Rows: {len(weekly):,}, groups: {weekly['group_id'].nunique():,}")
    print(f"Week range: {weekly['week_idx'].min()} to {weekly['week_idx'].max()}")
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
