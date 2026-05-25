"""
Independent weekly aggregate benchmark: LightGBM vs TFT.

This is not an M5 submission model. It compares models on the same weekly
aggregate forecasting task:

    aggregate group history -> next 4 weekly aggregate sales

The LightGBM benchmark uses four direct horizon models. Lag features are
computed relative to the forecast origin, not the target week, so validation
week sales are not used as features for later validation weeks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


DATA_DIR = Path("data")
REPORTS_DIR = Path("reports")
VALID_START_D = 1914
PREDICTION_WEEKS = 4

BASE_FEATURES = [
    "target_time_idx",
    "target_month",
    "target_year",
    "target_snap_days",
    "target_event_days",
    "target_christmas_days",
    "target_sell_price_mean",
    "origin_sales",
    "origin_log_sales",
    "lag_1",
    "lag_2",
    "lag_4",
    "lag_8",
    "lag_52",
    "rolling_mean_4",
    "rolling_mean_8",
    "rolling_mean_13",
    "rolling_std_8",
    "group_enc",
]


def parse_group_cols(value: str) -> list[str]:
    cols = [c.strip() for c in value.split(",") if c.strip()]
    allowed = {"state_id", "store_id", "cat_id", "dept_id", "item_id"}
    unknown = set(cols) - allowed
    if unknown:
        raise ValueError(f"Unsupported group columns: {sorted(unknown)}")
    if not cols:
        raise ValueError("At least one group column is required")
    return cols


def weekly_path(group_cols: list[str]) -> Path:
    suffix = "_".join(c.replace("_id", "") for c in group_cols)
    return DATA_DIR / f"tft_weekly_{suffix}_aligned.parquet"


def prediction_path(group_cols: list[str], variant: str) -> Path:
    suffix = "_".join(c.replace("_id", "") for c in group_cols)
    if variant == "tft_q50":
        return DATA_DIR / f"tft_weekly_{suffix}_revenue_lag_q50.csv"
    if variant == "tft_q75":
        return DATA_DIR / f"tft_weekly_{suffix}_revenue_lag_q75.csv"
    if variant == "tft_initial":
        return DATA_DIR / f"tft_weekly_{suffix}_predictions.csv"
    raise ValueError(f"Unknown variant: {variant}")


def add_history_features(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("time_idx").copy()
    sales = group["sales"].astype(float)
    group["origin_sales"] = sales
    group["origin_log_sales"] = np.log1p(sales)
    for lag in [1, 2, 4, 8, 52]:
        group[f"lag_{lag}"] = sales.shift(lag)
    group["rolling_mean_4"] = sales.shift(1).rolling(4).mean()
    group["rolling_mean_8"] = sales.shift(1).rolling(8).mean()
    group["rolling_mean_13"] = sales.shift(1).rolling(13).mean()
    group["rolling_std_8"] = sales.shift(1).rolling(8).std()
    return group


def build_direct_frame(weekly: pd.DataFrame, group_cols: list[str]) -> tuple[pd.DataFrame, int]:
    weekly = weekly.sort_values(["group_id", "time_idx"]).copy()
    weekly = weekly.groupby("group_id", group_keys=False).apply(add_history_features)
    weekly["group_enc"] = weekly["group_id"].astype("category").cat.codes.astype(np.int16)
    first_valid = int(weekly.loc[weekly["week_start_d"] == VALID_START_D, "time_idx"].min())

    target_cols = [
        "time_idx",
        "month",
        "year",
        "snap_days",
        "event_days",
        "christmas_days",
        "sell_price_mean",
        "sales",
        *group_cols,
    ]
    target = weekly[["group_id", *target_cols]].copy()
    target = target.rename(
        columns={
            "time_idx": "target_time_idx",
            "month": "target_month",
            "year": "target_year",
            "snap_days": "target_snap_days",
            "event_days": "target_event_days",
            "christmas_days": "target_christmas_days",
            "sell_price_mean": "target_sell_price_mean",
            "sales": "target_sales",
        }
    )

    frames = []
    origin_cols = [
        "group_id",
        "time_idx",
        "group_enc",
        "origin_sales",
        "origin_log_sales",
        "lag_1",
        "lag_2",
        "lag_4",
        "lag_8",
        "lag_52",
        "rolling_mean_4",
        "rolling_mean_8",
        "rolling_mean_13",
        "rolling_std_8",
    ]
    origins = weekly[origin_cols].copy()
    for horizon in range(1, PREDICTION_WEEKS + 1):
        frame = origins.copy()
        frame["target_time_idx"] = frame["time_idx"] + horizon
        frame = frame.merge(target, on=["group_id", "target_time_idx"], how="inner")
        frame["horizon"] = horizon
        frames.append(frame)

    direct = pd.concat(frames, ignore_index=True)
    return direct, first_valid


def train_direct_lightgbm(direct: pd.DataFrame, first_valid: int) -> pd.DataFrame:
    params = {
        "objective": "tweedie",
        "tweedie_variance_power": 1.1,
        "metric": "rmse",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "verbose": -1,
        "seed": 42,
        "n_jobs": -1,
    }

    preds = []
    for horizon in range(1, PREDICTION_WEEKS + 1):
        train = direct[
            (direct["horizon"] == horizon)
            & (direct["target_time_idx"] < first_valid)
        ].copy()
        valid = direct[
            (direct["horizon"] == horizon)
            & (direct["target_time_idx"] == first_valid + horizon - 1)
        ].copy()

        train = train.dropna(subset=BASE_FEATURES + ["target_sales"])
        valid = valid.dropna(subset=BASE_FEATURES)

        dtrain = lgb.Dataset(
            train[BASE_FEATURES],
            label=train["target_sales"],
            weight=train["target_sales"].clip(lower=1).to_numpy() ** 0.25,
            free_raw_data=False,
        )
        model = lgb.train(params, dtrain, num_boost_round=400)
        valid[f"week_{horizon}"] = np.maximum(0.0, model.predict(valid[BASE_FEATURES]))
        preds.append(valid[["group_id", *[c for c in valid.columns if c in GROUP_COLS_GLOBAL], f"week_{horizon}"]])

    out = preds[0]
    for frame in preds[1:]:
        out = out.merge(frame, on=["group_id", *GROUP_COLS_GLOBAL], how="inner")
    return out[GROUP_COLS_GLOBAL + [f"week_{i}" for i in range(1, PREDICTION_WEEKS + 1)]]


def actual_weekly(weekly: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    valid = weekly[weekly["week_start_d"].between(VALID_START_D, VALID_START_D + 21)].copy()
    valid["week_num"] = ((valid["week_start_d"] - VALID_START_D) // 7 + 1).astype(int)
    rows = valid[group_cols].drop_duplicates().copy()
    for i in range(1, PREDICTION_WEEKS + 1):
        part = valid[valid["week_num"] == i][group_cols + ["sales"]]
        rows = rows.merge(part.rename(columns={"sales": f"week_{i}"}), on=group_cols, how="left")
    return rows


def load_tft(path: Path, group_cols: list[str]) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df[group_cols + [f"week_{i}" for i in range(1, PREDICTION_WEEKS + 1)]]


def weekly_scales(weekly: pd.DataFrame, group_cols: list[str], first_valid: int) -> pd.DataFrame:
    train = weekly[weekly["time_idx"] < first_valid].sort_values(group_cols + ["time_idx"]).copy()

    def scale_one(group: pd.DataFrame) -> float:
        diff = group["sales"].astype(float).diff().dropna().to_numpy()
        if len(diff) == 0:
            return 1.0
        scale = float(np.mean(diff ** 2))
        return max(scale, 1e-9)

    scales = train.groupby(group_cols).apply(scale_one).reset_index(name="scale")
    return scales


def metrics(
    pred: pd.DataFrame,
    actual: pd.DataFrame,
    group_cols: list[str],
    scales: pd.DataFrame,
) -> dict[str, float]:
    merged = actual.merge(pred, on=group_cols, suffixes=("_actual", "_pred"))
    merged = merged.merge(scales, on=group_cols, how="left")
    actual_arr = merged[[f"week_{i}_actual" for i in range(1, PREDICTION_WEEKS + 1)]].to_numpy(float)
    pred_arr = merged[[f"week_{i}_pred" for i in range(1, PREDICTION_WEEKS + 1)]].to_numpy(float)
    err = pred_arr - actual_arr
    denom = max(float(np.abs(actual_arr).sum()), 1e-9)
    scale = merged["scale"].fillna(1.0).to_numpy(float)
    group_mse = np.mean(err ** 2, axis=1)
    rmsse = np.sqrt(group_mse / np.maximum(scale, 1e-9))

    return {
        "groups": int(len(merged)),
        "wape": float(np.abs(err).sum() / denom),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "rmsse": float(np.mean(rmsse)),
        "bias": float(err.sum() / denom),
    }


def run_level(group_cols: list[str]) -> pd.DataFrame:
    weekly = pd.read_parquet(weekly_path(group_cols))
    direct, first_valid = build_direct_frame(weekly, group_cols)
    lgb_pred = train_direct_lightgbm(direct, first_valid)
    actual = actual_weekly(weekly, group_cols)
    scales = weekly_scales(weekly, group_cols, first_valid)

    rows = []
    comparisons = {"weekly_lgbm_direct": lgb_pred}
    for variant in ["tft_initial", "tft_q50", "tft_q75"]:
        pred = load_tft(prediction_path(group_cols, variant), group_cols)
        if pred is not None:
            comparisons[variant] = pred

    for name, pred in comparisons.items():
        m = metrics(pred, actual, group_cols, scales)
        rows.append(
            {
                "level": ",".join(group_cols),
                "model": name,
                **m,
            }
        )

    suffix = "_".join(c.replace("_id", "") for c in group_cols)
    out_path = DATA_DIR / f"weekly_lgbm_{suffix}_direct_predictions.csv"
    lgb_pred.to_csv(out_path, index=False)
    return pd.DataFrame(rows)


GROUP_COLS_GLOBAL: list[str] = []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--levels",
        default="store_id,cat_id;store_id,dept_id",
        help="Semicolon-separated group levels.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "weekly_lgbm_vs_tft_benchmark.csv",
    )
    args = parser.parse_args()

    all_results = []
    for level in args.levels.split(";"):
        group_cols = parse_group_cols(level)
        global GROUP_COLS_GLOBAL
        GROUP_COLS_GLOBAL = group_cols
        print(f"Running weekly benchmark for {group_cols}")
        all_results.append(run_level(group_cols))

    results = pd.concat(all_results, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)

    print("\nWeekly benchmark results")
    print(results.to_string(index=False, formatters={
        "wape": "{:.6f}".format,
        "rmse": "{:.3f}".format,
        "rmsse": "{:.6f}".format,
        "bias": "{:.3%}".format,
    }))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
