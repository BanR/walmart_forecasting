"""
Build a non-leaking weekly aggregate baseline forecast.

This creates the same file shape expected by src.ensemble.weekly_reconcile:

    store_id,cat_id,week_1,week_2,week_3,week_4

It is useful for testing the weekly aggregate reconciliation pipeline before a
TFT/N-BEATS weekly model has produced forecasts.

Default method:
    avg_last_4_matching_weeks

For each aggregate group, forecast each future week as the average of the last
four observed weekly totals for that group. Since the validation horizon is
d_1914..d_1941, the observed history used here ends at d_1913.

Usage:

    python -m src.ensemble.build_weekly_baseline_forecast \
      --group-cols store_id,cat_id \
      --output data/weekly_store_cat_baseline_predictions.csv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("data")
DEFAULT_OUTPUT = DATA_DIR / "weekly_store_cat_baseline_predictions.csv"
TRAIN_END_DAY = 1913
HORIZON_WEEKS = 4


def parse_group_cols(value: str) -> list[str]:
    cols = [c.strip() for c in value.split(",") if c.strip()]
    allowed = {"state_id", "store_id", "cat_id", "dept_id", "item_id"}
    unknown = set(cols) - allowed
    if unknown:
        raise ValueError(f"Unsupported group columns: {sorted(unknown)}")
    if not cols:
        raise ValueError("At least one group column is required")
    return cols


def load_weekly_history(group_cols: list[str]) -> pd.DataFrame:
    """Aggregate d_1..d_1913 to weekly group totals."""
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [f"d_{i}" for i in range(1, TRAIN_END_DAY + 1)]
    sales = pd.read_csv(DATA_DIR / "sales_train_evaluation.csv", usecols=id_cols + day_cols)

    daily = sales[id_cols + day_cols].melt(
        id_vars=id_cols,
        var_name="d",
        value_name="sales",
    )
    daily["d_num"] = daily["d"].str.replace("d_", "", regex=False).astype(int)
    daily["week_idx"] = ((daily["d_num"] - 1) // 7).astype(int)

    weekly = daily.groupby(group_cols + ["week_idx"], as_index=False)["sales"].sum()
    return weekly


def build_forecast(
    weekly: pd.DataFrame,
    group_cols: list[str],
    lookback_weeks: int,
) -> pd.DataFrame:
    """Forecast each horizon week with the recent weekly average per group."""
    last_week = weekly["week_idx"].max()
    recent = weekly[weekly["week_idx"].between(last_week - lookback_weeks + 1, last_week)]

    base = recent.groupby(group_cols, as_index=False)["sales"].mean()
    base = base.rename(columns={"sales": "weekly_forecast"})

    rows = base[group_cols].copy()
    for i in range(1, HORIZON_WEEKS + 1):
        rows[f"week_{i}"] = base["weekly_forecast"].to_numpy(dtype=np.float32)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-cols", default="store_id,cat_id")
    parser.add_argument("--lookback-weeks", type=int, default=4)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.lookback_weeks < 1:
        raise ValueError("--lookback-weeks must be >= 1")

    t0 = time.time()
    group_cols = parse_group_cols(args.group_cols)
    print(f"Building weekly baseline forecast for group_cols={group_cols}")

    weekly = load_weekly_history(group_cols)
    forecast = build_forecast(weekly, group_cols, args.lookback_weeks)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(args.output, index=False)

    print(f"Saved: {args.output}")
    print(f"Rows: {len(forecast):,}")
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
