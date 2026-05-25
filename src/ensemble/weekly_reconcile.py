"""
Reconcile daily bottom-level LightGBM forecasts with weekly aggregate forecasts.

This is the bridge for the hybrid approach:

1. LightGBM predicts daily item-store sales.
2. TFT or another sequence model predicts weekly aggregate sales, for example
   store-category-week or store-department-week.
3. This script aggregates LightGBM to the same weekly level, computes a
   shrunk correction ratio, and applies it back to the daily item-store rows.

The aggregate forecast file should contain one row per aggregate group and
four weekly forecast columns. Supported week column names:

    week_1, week_2, week_3, week_4

or:

    F1, F2, F3, F4

Example:

    store_id,dept_id,week_1,week_2,week_3,week_4
    CA_1,FOODS_3,12345.0,12011.2,11890.4,13001.5

Usage:

    python -m src.ensemble.weekly_reconcile \
      --base-preds data/predictions_lgb_v2.parquet \
      --agg-preds data/tft_weekly_store_dept_revenue_lag_q75.csv \
      --group-cols store_id,dept_id \
      --alpha 0.10 \
      --output data/predictions_lgb_v2_weekly_tft_store_dept_revenue_lag_q75_a010.parquet \
      --score

Use --oracle-debug only for validation experiments. It uses actual validation
sales as the aggregate forecast and therefore must never be used for a real
submission.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics.wrmsse import WRMSSEEvaluator


DATA_DIR = Path("data")
DEFAULT_BASE_PREDS = DATA_DIR / "predictions_lgb_v2.parquet"
DEFAULT_AGG_PREDS = DATA_DIR / "tft_weekly_store_dept_revenue_lag_q75.csv"
DEFAULT_OUTPUT = DATA_DIR / "predictions_lgb_v2_weekly_tft_store_dept_revenue_lag_q75_a010.parquet"
VALID_DAY_COLS = [f"d_{i}" for i in range(1914, 1942)]
WEEK_COLS = [f"week_{i}" for i in range(1, 5)]


def parse_group_cols(value: str) -> list[str]:
    """Parse comma-separated grouping columns."""
    cols = [c.strip() for c in value.split(",") if c.strip()]
    allowed = {"state_id", "store_id", "cat_id", "dept_id", "item_id"}
    unknown = set(cols) - allowed
    if unknown:
        raise ValueError(f"Unsupported group columns: {sorted(unknown)}")
    if not cols:
        raise ValueError("At least one group column is required")
    return cols


def load_sales_metadata() -> pd.DataFrame:
    """Load bottom-level identifiers in the canonical M5 row order."""
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    return pd.read_csv(DATA_DIR / "sales_train_evaluation.csv", usecols=id_cols)


def load_base_predictions(path: Path) -> pd.DataFrame:
    """Load daily bottom-level predictions and normalize to d_1914..d_1941 columns."""
    preds = pd.read_parquet(path)
    if list(preds.columns) == VALID_DAY_COLS:
        return preds.copy()

    if len(preds.columns) == 28:
        out = preds.copy()
        out.columns = VALID_DAY_COLS
        return out

    missing = [c for c in VALID_DAY_COLS if c not in preds.columns]
    if missing:
        raise ValueError(f"Prediction file is missing validation columns: {missing[:5]}")
    return preds[VALID_DAY_COLS].copy()


def add_week_columns(df: pd.DataFrame, day_cols: list[str]) -> pd.DataFrame:
    """Add four weekly totals from 28 daily columns."""
    out = df.copy()
    for i in range(4):
        cols = day_cols[i * 7 : (i + 1) * 7]
        out[WEEK_COLS[i]] = out[cols].sum(axis=1)
    return out


def aggregate_weekly(
    bottom_df: pd.DataFrame,
    group_cols: list[str],
    week_cols: list[str] = WEEK_COLS,
) -> pd.DataFrame:
    """Aggregate bottom-level weekly values to the selected hierarchy."""
    return bottom_df.groupby(group_cols, as_index=False)[week_cols].sum()


def load_aggregate_forecasts(path: Path, group_cols: list[str]) -> pd.DataFrame:
    """Load TFT/aggregate weekly forecasts."""
    if not path.exists():
        raise FileNotFoundError(
            f"Aggregate forecast file not found: {path}\n"
            "Create this file from a weekly aggregate TFT/N-BEATS model first, "
            "or use --oracle-debug for a validation-only upper-bound check.\n"
            "Expected schema example:\n"
            "  store_id,dept_id,week_1,week_2,week_3,week_4\n"
            "  CA_1,FOODS_3,12345.0,12011.2,11890.4,13001.5"
        )

    if path.suffix.lower() == ".parquet":
        agg = pd.read_parquet(path)
    else:
        agg = pd.read_csv(path)

    rename = {f"F{i}": f"week_{i}" for i in range(1, 5)}
    agg = agg.rename(columns=rename)

    missing_groups = [c for c in group_cols if c not in agg.columns]
    missing_weeks = [c for c in WEEK_COLS if c not in agg.columns]
    if missing_groups or missing_weeks:
        raise ValueError(
            "Aggregate forecast file has wrong schema. "
            f"Missing groups={missing_groups}, missing weeks={missing_weeks}"
        )

    return agg[group_cols + WEEK_COLS].copy()


def build_oracle_aggregate_forecasts(group_cols: list[str]) -> pd.DataFrame:
    """
    Build aggregate forecasts from validation actuals.

    This is useful only to estimate the upper bound of the reconciliation
    method. It leaks validation labels and must not be used for submission.
    """
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    sales = pd.read_csv(DATA_DIR / "sales_train_evaluation.csv", usecols=id_cols + VALID_DAY_COLS)
    actual = add_week_columns(sales, VALID_DAY_COLS)
    return aggregate_weekly(actual, group_cols)


def compute_ratios(
    base_weekly: pd.DataFrame,
    target_weekly: pd.DataFrame,
    group_cols: list[str],
    alpha: float,
    min_ratio: float,
    max_ratio: float,
    epsilon: float,
) -> pd.DataFrame:
    """
    Compute shrunk ratios:

        adjusted_ratio = 1 + alpha * (target / base - 1)
    """
    merged = base_weekly.merge(
        target_weekly,
        on=group_cols,
        how="left",
        suffixes=("_base", "_target"),
    )

    ratio_df = merged[group_cols].copy()
    for week_col in WEEK_COLS:
        base = merged[f"{week_col}_base"].to_numpy(dtype=np.float64)
        target = merged[f"{week_col}_target"].to_numpy(dtype=np.float64)

        raw_ratio = np.ones_like(base)
        valid = np.isfinite(target) & (base > epsilon)
        raw_ratio[valid] = target[valid] / base[valid]

        adjusted = 1.0 + alpha * (raw_ratio - 1.0)
        adjusted = np.clip(adjusted, min_ratio, max_ratio)

        # If the aggregate model has no row for this group, leave base unchanged.
        adjusted[~np.isfinite(target)] = 1.0
        ratio_df[f"{week_col}_ratio"] = adjusted.astype(np.float32)

    return ratio_df


def apply_weekly_ratios(
    base_daily: pd.DataFrame,
    metadata: pd.DataFrame,
    ratios: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    """Apply aggregate weekly ratios back to daily bottom-level forecasts."""
    daily = pd.concat(
        [metadata[group_cols].reset_index(drop=True), base_daily.reset_index(drop=True)],
        axis=1,
    )
    daily = daily.merge(ratios, on=group_cols, how="left")

    adjusted = base_daily.copy()
    for i in range(4):
        day_cols = VALID_DAY_COLS[i * 7 : (i + 1) * 7]
        ratio_col = f"week_{i + 1}_ratio"
        ratios_i = daily[ratio_col].fillna(1.0).to_numpy(dtype=np.float32)
        adjusted.loc[:, day_cols] = adjusted[day_cols].multiply(ratios_i, axis=0)

    return adjusted.clip(lower=0)


def score_predictions(preds: pd.DataFrame) -> tuple[float, dict[str, float]]:
    """Score predictions with the existing WRMSSE evaluator."""
    sales = pd.read_csv(DATA_DIR / "sales_train_evaluation.csv")
    calendar = pd.read_csv(DATA_DIR / "calendar.csv")
    prices = pd.read_csv(DATA_DIR / "sell_prices.csv")

    id_cols = [c for c in sales.columns if not c.startswith("d_")]
    train_df = sales[id_cols + [f"d_{i}" for i in range(1, 1914)]]
    valid_df = sales[id_cols + VALID_DAY_COLS]

    evaluator = WRMSSEEvaluator(
        train_df=train_df,
        valid_df=valid_df,
        calendar=calendar,
        prices=prices,
    )
    return evaluator.score(preds), evaluator.score_per_level(preds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-preds", type=Path, default=DEFAULT_BASE_PREDS)
    parser.add_argument("--agg-preds", type=Path, default=DEFAULT_AGG_PREDS)
    parser.add_argument("--group-cols", default="store_id,dept_id")
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--min-ratio", type=float, default=0.85)
    parser.add_argument("--max-ratio", type=float, default=1.15)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--score", action="store_true", help="Compute WRMSSE before and after correction")
    parser.add_argument(
        "--oracle-debug",
        action="store_true",
        help="Use validation actuals as aggregate forecasts to estimate upper bound. Leaks labels.",
    )
    args = parser.parse_args()

    if not 0 <= args.alpha <= 1:
        raise ValueError("--alpha must be between 0 and 1")

    t0 = time.time()
    group_cols = parse_group_cols(args.group_cols)
    metadata = load_sales_metadata()
    base_daily = load_base_predictions(args.base_preds)

    bottom = pd.concat([metadata.reset_index(drop=True), base_daily.reset_index(drop=True)], axis=1)
    base_weekly = aggregate_weekly(add_week_columns(bottom, VALID_DAY_COLS), group_cols)

    if args.oracle_debug:
        print("WARNING: --oracle-debug leaks validation labels. Use only for diagnostics.")
        target_weekly = build_oracle_aggregate_forecasts(group_cols)
    else:
        target_weekly = load_aggregate_forecasts(args.agg_preds, group_cols)

    ratios = compute_ratios(
        base_weekly=base_weekly,
        target_weekly=target_weekly,
        group_cols=group_cols,
        alpha=args.alpha,
        min_ratio=args.min_ratio,
        max_ratio=args.max_ratio,
        epsilon=args.epsilon,
    )
    adjusted = apply_weekly_ratios(base_daily, metadata, ratios, group_cols)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    adjusted.to_parquet(args.output, index=False)

    print(f"Saved reconciled predictions: {args.output}")
    print(f"Group level: {group_cols}")
    print(f"Alpha={args.alpha}, ratio clip=[{args.min_ratio}, {args.max_ratio}]")
    print(f"Ratio summary:\n{ratios[[c for c in ratios.columns if c.endswith('_ratio')]].describe()}")

    if args.score:
        base_score, _ = score_predictions(base_daily)
        adjusted_score, per_level = score_predictions(adjusted)
        print("\nWRMSSE")
        print(f"  base:     {base_score:.6f}")
        print(f"  adjusted: {adjusted_score:.6f}")
        print("\nAdjusted per-level:")
        for key, value in per_level.items():
            print(f"  {key}: {value:.6f}")

    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
