"""
Evaluate weekly aggregate forecasts against LightGBM rollups.

Use this before applying weekly_reconcile. If the aggregate model is worse than
the LightGBM rollup at the same weekly hierarchy, reconciliation is likely to
hurt full WRMSSE.

This is a weekly diagnostic, not the official M5 WRMSSE. It reports WAPE,
RMSE, bias, and weekly RMSSE. Weekly RMSSE below 1 means the aggregate forecast
beats a weekly naive benchmark under this aggregate weekly definition.

Usage:

    python -m src.ensemble.evaluate_weekly_aggregate \
      --base-preds data/predictions_lgb_v2.parquet \
      --agg-preds data/weekly_store_cat_baseline_predictions.csv \
      --group-cols store_id,cat_id
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("data")
VALID_DAY_COLS = [f"d_{i}" for i in range(1914, 1942)]
WEEK_COLS = [f"week_{i}" for i in range(1, 5)]


def parse_group_cols(value: str) -> list[str]:
    cols = [c.strip() for c in value.split(",") if c.strip()]
    allowed = {"state_id", "store_id", "cat_id", "dept_id", "item_id"}
    unknown = set(cols) - allowed
    if unknown:
        raise ValueError(f"Unsupported group columns: {sorted(unknown)}")
    if not cols:
        raise ValueError("At least one group column is required")
    return cols


def load_base_predictions(path: Path) -> pd.DataFrame:
    preds = pd.read_parquet(path)
    if list(preds.columns) == VALID_DAY_COLS:
        return preds.copy()
    if len(preds.columns) == 28:
        out = preds.copy()
        out.columns = VALID_DAY_COLS
        return out
    return preds[VALID_DAY_COLS].copy()


def load_aggregate_predictions(path: Path, group_cols: list[str]) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        agg = pd.read_parquet(path)
    else:
        agg = pd.read_csv(path)
    agg = agg.rename(columns={f"F{i}": f"week_{i}" for i in range(1, 5)})
    return agg[group_cols + WEEK_COLS].copy()


def add_week_columns(df: pd.DataFrame, day_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for i in range(4):
        out[WEEK_COLS[i]] = out[day_cols[i * 7 : (i + 1) * 7]].sum(axis=1)
    return out


def aggregate_weekly(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return df.groupby(group_cols, as_index=False)[WEEK_COLS].sum()


def weekly_scales(group_cols: list[str]) -> pd.DataFrame:
    """Compute weekly naive scaling denominator from pre-validation history."""
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    train_day_cols = [f"d_{i}" for i in range(1, 1914)]
    sales = pd.read_csv(DATA_DIR / "sales_train_evaluation.csv", usecols=id_cols + train_day_cols)
    daily = sales.melt(id_vars=id_cols, var_name="d", value_name="sales")
    daily["d_num"] = daily["d"].str.replace("d_", "", regex=False).astype(int)
    daily["week_idx"] = ((daily["d_num"] - 1) // 7).astype(int)
    weekly = daily.groupby(group_cols + ["week_idx"], as_index=False)["sales"].sum()

    def scale_one(group: pd.DataFrame) -> float:
        diff = group.sort_values("week_idx")["sales"].astype(float).diff().dropna().to_numpy()
        if len(diff) == 0:
            return 1.0
        return max(float(np.mean(diff ** 2)), 1e-9)

    return weekly.groupby(group_cols).apply(scale_one, include_groups=False).reset_index(name="scale")


def metrics(pred: np.ndarray, actual: np.ndarray, scale: np.ndarray | None = None) -> dict[str, float]:
    err = pred - actual
    denom = np.maximum(np.abs(actual).sum(), 1e-9)
    out = {
        "wape": float(np.abs(err).sum() / denom),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "bias_pct": float(err.sum() / denom),
    }
    if scale is not None:
        group_mse = np.mean(err ** 2, axis=1)
        out["rmsse"] = float(np.mean(np.sqrt(group_mse / np.maximum(scale, 1e-9))))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-preds", type=Path, default=DATA_DIR / "predictions_lgb_v2.parquet")
    parser.add_argument("--agg-preds", type=Path, required=True)
    parser.add_argument("--group-cols", default="store_id,cat_id")
    args = parser.parse_args()

    group_cols = parse_group_cols(args.group_cols)
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    metadata = pd.read_csv(DATA_DIR / "sales_train_evaluation.csv", usecols=id_cols)
    actual_daily = pd.read_csv(DATA_DIR / "sales_train_evaluation.csv", usecols=id_cols + VALID_DAY_COLS)

    base_daily = load_base_predictions(args.base_preds)
    base_bottom = pd.concat([metadata.reset_index(drop=True), base_daily.reset_index(drop=True)], axis=1)

    actual_weekly = aggregate_weekly(add_week_columns(actual_daily, VALID_DAY_COLS), group_cols)
    base_weekly = aggregate_weekly(add_week_columns(base_bottom, VALID_DAY_COLS), group_cols)
    agg_weekly = load_aggregate_predictions(args.agg_preds, group_cols)
    scales = weekly_scales(group_cols)

    merged = actual_weekly.merge(base_weekly, on=group_cols, suffixes=("_actual", "_base"))
    merged = merged.merge(agg_weekly, on=group_cols, how="left")
    merged = merged.merge(scales, on=group_cols, how="left")

    actual = merged[[f"{c}_actual" for c in WEEK_COLS]].to_numpy(dtype=float)
    base = merged[[f"{c}_base" for c in WEEK_COLS]].to_numpy(dtype=float)
    agg = merged[WEEK_COLS].to_numpy(dtype=float)
    scale = merged["scale"].fillna(1.0).to_numpy(dtype=float)

    base_metrics = metrics(base, actual, scale)
    agg_metrics = metrics(agg, actual, scale)

    print(f"Group level: {group_cols}")
    print(f"Groups: {len(merged):,}")
    print("\nAggregate validation metrics")
    print("Lower WAPE/RMSE/RMSSE is better. RMSSE < 1 beats weekly naive. Bias near 0 is better.")
    print(
        "  LightGBM rollup: "
        f"RMSSE={base_metrics['rmsse']:.6f}, WAPE={base_metrics['wape']:.6f}, "
        f"RMSE={base_metrics['rmse']:.3f}, bias={base_metrics['bias_pct']:.3%}"
    )
    print(
        "  Aggregate model: "
        f"RMSSE={agg_metrics['rmsse']:.6f}, WAPE={agg_metrics['wape']:.6f}, "
        f"RMSE={agg_metrics['rmse']:.3f}, bias={agg_metrics['bias_pct']:.3%}"
    )

    if agg_metrics["rmsse"] < base_metrics["rmsse"]:
        print("\nAggregate model beats the LightGBM rollup on weekly RMSSE. Reconciliation may help.")
    else:
        print("\nAggregate model is worse than the LightGBM rollup on weekly RMSSE. Use only as a shrunk correction signal.")


if __name__ == "__main__":
    main()
