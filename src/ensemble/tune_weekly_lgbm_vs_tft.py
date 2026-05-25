"""
Tune the independent weekly aggregate benchmark.

This script tunes:
1. weekly LightGBM direct models on a compact parameter grid;
2. TFT point-quantile selection from already exported quantile forecasts.

It reports comparable weekly aggregate metrics at the same levels.
The primary selection metric is weekly RMSSE, so the output aligns with the
review dashboard and experiment report. WAPE/RMSE remain diagnostic columns.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


DATA_DIR = Path("data")
REPORTS_DIR = Path("reports")
VALID_START_D = 1914
PREDICTION_WEEKS = 4
QUANTILES = [0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98]


def weekly_path(group_cols: list[str]) -> Path:
    suffix = "_".join(c.replace("_id", "") for c in group_cols)
    return DATA_DIR / f"tft_weekly_{suffix}_aligned.parquet"


def tft_quantile_path(group_cols: list[str]) -> Path:
    suffix = "_".join(c.replace("_id", "") for c in group_cols)
    return DATA_DIR / f"tft_weekly_{suffix}_revenue_lag_quantiles.csv"


def add_history_features(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("time_idx").copy()
    sales = group["sales"].astype(float)
    group["origin_sales"] = sales
    group["origin_log_sales"] = np.log1p(sales)
    for lag in [1, 2, 4, 8, 13, 26, 52]:
        group[f"lag_{lag}"] = sales.shift(lag)
    for window in [4, 8, 13, 26, 52]:
        shifted = sales.shift(1)
        group[f"rolling_mean_{window}"] = shifted.rolling(window).mean()
        group[f"rolling_std_{window}"] = shifted.rolling(window).std()
    return group


def prepare_direct_frame(weekly: pd.DataFrame, group_cols: list[str]) -> tuple[pd.DataFrame, int, list[str]]:
    weekly = weekly.sort_values(["group_id", "time_idx"]).copy()
    weekly = pd.concat(
        [add_history_features(group) for _, group in weekly.groupby("group_id", sort=False)],
        ignore_index=True,
    )
    weekly["group_enc"] = weekly["group_id"].astype("category").cat.codes.astype(np.int16)
    first_valid = int(weekly.loc[weekly["week_start_d"] == VALID_START_D, "time_idx"].min())

    target = weekly[
        [
            "group_id",
            *group_cols,
            "time_idx",
            "month",
            "year",
            "snap_days",
            "event_days",
            "christmas_days",
            "sell_price_mean",
            "sales",
        ]
    ].copy()
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

    feature_cols = [
        "target_time_idx",
        "target_month",
        "target_year",
        "target_snap_days",
        "target_event_days",
        "target_christmas_days",
        "target_sell_price_mean",
        "origin_sales",
        "origin_log_sales",
        "group_enc",
        *[f"lag_{lag}" for lag in [1, 2, 4, 8, 13, 26, 52]],
        *[f"rolling_mean_{w}" for w in [4, 8, 13, 26, 52]],
        *[f"rolling_std_{w}" for w in [4, 8, 13, 26, 52]],
    ]

    origins = weekly[["group_id", "time_idx", *feature_cols[7:]]].copy()
    frames = []
    for horizon in range(1, PREDICTION_WEEKS + 1):
        frame = origins.copy()
        frame["target_time_idx"] = frame["time_idx"] + horizon
        frame = frame.merge(target, on=["group_id", "target_time_idx"], how="inner")
        frame["horizon"] = horizon
        frames.append(frame)

    direct = pd.concat(frames, ignore_index=True)
    return direct, first_valid, feature_cols


def actual_weekly(weekly: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    valid = weekly[weekly["week_start_d"].between(VALID_START_D, VALID_START_D + 21)].copy()
    valid["week_num"] = ((valid["week_start_d"] - VALID_START_D) // 7 + 1).astype(int)
    out = valid[group_cols].drop_duplicates().copy()
    for i in range(1, PREDICTION_WEEKS + 1):
        part = valid[valid["week_num"] == i][group_cols + ["sales"]]
        out = out.merge(part.rename(columns={"sales": f"week_{i}"}), on=group_cols, how="left")
    return out


def weekly_scales(weekly: pd.DataFrame, group_cols: list[str], first_valid: int) -> pd.DataFrame:
    train = weekly[weekly["time_idx"] < first_valid].sort_values(group_cols + ["time_idx"]).copy()

    def scale_one(group: pd.DataFrame) -> float:
        diff = group["sales"].astype(float).diff().dropna().to_numpy()
        return max(float(np.mean(diff ** 2)), 1e-9) if len(diff) else 1.0

    return train.groupby(group_cols).apply(scale_one, include_groups=False).reset_index(name="scale")


def metrics(pred: pd.DataFrame, actual: pd.DataFrame, scales: pd.DataFrame, group_cols: list[str]) -> dict[str, float]:
    merged = actual.merge(pred, on=group_cols, suffixes=("_actual", "_pred"))
    merged = merged.merge(scales, on=group_cols, how="left")
    actual_arr = merged[[f"week_{i}_actual" for i in range(1, 5)]].to_numpy(float)
    pred_arr = merged[[f"week_{i}_pred" for i in range(1, 5)]].to_numpy(float)
    err = pred_arr - actual_arr
    denom = max(float(np.abs(actual_arr).sum()), 1e-9)
    scale = merged["scale"].fillna(1.0).to_numpy(float)
    group_mse = np.mean(err ** 2, axis=1)
    return {
        "wape": float(np.abs(err).sum() / denom),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "rmsse": float(np.mean(np.sqrt(group_mse / np.maximum(scale, 1e-9)))),
        "bias": float(err.sum() / denom),
    }


def train_lgbm_combo(
    direct: pd.DataFrame,
    first_valid: int,
    group_cols: list[str],
    feature_cols: list[str],
    combo: dict[str, object],
) -> pd.DataFrame:
    frames = []
    for horizon in range(1, PREDICTION_WEEKS + 1):
        train = direct[(direct["horizon"] == horizon) & (direct["target_time_idx"] < first_valid)].copy()
        valid = direct[
            (direct["horizon"] == horizon)
            & (direct["target_time_idx"] == first_valid + horizon - 1)
        ].copy()
        train = train.dropna(subset=feature_cols + ["target_sales"])
        valid = valid.dropna(subset=feature_cols)

        params = {
            "objective": combo["objective"],
            "metric": "rmse",
            "learning_rate": combo["learning_rate"],
            "num_leaves": combo["num_leaves"],
            "min_data_in_leaf": combo["min_data_in_leaf"],
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 1,
            "verbose": -1,
            "seed": 42,
            "num_threads": 4,
            "force_row_wise": True,
        }
        if combo["objective"] == "tweedie":
            params["tweedie_variance_power"] = combo["tweedie_variance_power"]

        if combo["weight"] == "none":
            weight = None
        elif combo["weight"] == "sales_025":
            weight = train["target_sales"].clip(lower=1).to_numpy(float) ** 0.25
        elif combo["weight"] == "sales_050":
            weight = train["target_sales"].clip(lower=1).to_numpy(float) ** 0.50
        else:
            raise ValueError(combo["weight"])

        dataset = lgb.Dataset(train[feature_cols], label=train["target_sales"], weight=weight, free_raw_data=False)
        model = lgb.train(params, dataset, num_boost_round=combo["rounds"])
        valid[f"week_{horizon}"] = np.maximum(0.0, model.predict(valid[feature_cols]))
        frames.append(valid[["group_id", *group_cols, f"week_{horizon}"]])

    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on=["group_id", *group_cols], how="inner")
    return out[group_cols + [f"week_{i}" for i in range(1, 5)]]


def tft_quantile_forecasts(qdf: pd.DataFrame, group_cols: list[str]) -> dict[str, pd.DataFrame]:
    out = {}
    for q in QUANTILES:
        suffix = f"q{str(q).replace('.', '')}"
        frame = qdf[group_cols].copy()
        for i in range(1, 5):
            frame[f"week_{i}"] = qdf[f"week_{i}_{suffix}"]
        out[f"q{q:.2f}"] = frame
    return out


def tune_level(group_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    weekly = pd.read_parquet(weekly_path(group_cols))
    actual = actual_weekly(weekly, group_cols)
    direct, first_valid, feature_cols = prepare_direct_frame(weekly, group_cols)
    scales = weekly_scales(weekly, group_cols, first_valid)

    lgbm_grid = [
        {
            "objective": "regression",
            "tweedie_variance_power": None,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 20,
            "rounds": 600,
            "weight": "none",
        },
        {
            "objective": "regression",
            "tweedie_variance_power": None,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 20,
            "rounds": 600,
            "weight": "sales_025",
        },
        {
            "objective": "regression",
            "tweedie_variance_power": None,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_data_in_leaf": 20,
            "rounds": 600,
            "weight": "sales_025",
        },
        {
            "objective": "tweedie",
            "tweedie_variance_power": 1.1,
            "learning_rate": 0.01,
            "num_leaves": 31,
            "min_data_in_leaf": 20,
            "rounds": 600,
            "weight": "none",
        },
        {
            "objective": "tweedie",
            "tweedie_variance_power": 1.1,
            "learning_rate": 0.01,
            "num_leaves": 31,
            "min_data_in_leaf": 20,
            "rounds": 600,
            "weight": "sales_025",
        },
        {
            "objective": "tweedie",
            "tweedie_variance_power": 1.3,
            "learning_rate": 0.01,
            "num_leaves": 31,
            "min_data_in_leaf": 20,
            "rounds": 600,
            "weight": "sales_025",
        },
    ]

    lgbm_rows = []
    best_pred = None
    best_score = float("inf")
    best_name = ""
    for i, combo in enumerate(lgbm_grid, start=1):
        pred = train_lgbm_combo(direct, first_valid, group_cols, feature_cols, combo)
        m = metrics(pred, actual, scales, group_cols)
        name = (
            f"{combo['objective']}_p{combo['tweedie_variance_power']}_"
            f"lr{combo['learning_rate']}_leaves{combo['num_leaves']}_"
            f"leaf{combo['min_data_in_leaf']}_r{combo['rounds']}_{combo['weight']}"
        )
        lgbm_rows.append({"level": ",".join(group_cols), "model": "weekly_lgbm", "name": name, **combo, **m})
        print(
            f"  {i}/{len(lgbm_grid)} {name}: "
            f"WAPE={m['wape']:.6f}, RMSE={m['rmse']:.3f}, RMSSE={m['rmsse']:.6f}",
            flush=True,
        )
        if m["rmsse"] < best_score:
            best_score = m["rmsse"]
            best_pred = pred
            best_name = name

    suffix = "_".join(c.replace("_id", "") for c in group_cols)
    best_pred.to_csv(DATA_DIR / f"weekly_lgbm_{suffix}_tuned_best_predictions.csv", index=False)

    tft_rows = []
    qpath = tft_quantile_path(group_cols)
    if qpath.exists():
        qdf = pd.read_csv(qpath)
        for name, pred in tft_quantile_forecasts(qdf, group_cols).items():
            tft_rows.append({"level": ",".join(group_cols), "model": "weekly_tft", "name": name, **metrics(pred, actual, scales, group_cols)})

    lgbm_df = pd.DataFrame(lgbm_rows).sort_values(["level", "rmsse", "wape", "rmse"]).reset_index(drop=True)
    tft_df = pd.DataFrame(tft_rows).sort_values(["level", "rmsse", "wape", "rmse"]).reset_index(drop=True)
    print(f"Best LightGBM for {group_cols}: {best_name} RMSSE={best_score:.6f}")
    return lgbm_df, tft_df


def main() -> None:
    levels = [["store_id", "cat_id"], ["store_id", "dept_id"]]
    lgbm_all = []
    tft_all = []
    for group_cols in levels:
        print(f"\nTuning level: {group_cols}")
        lgbm_df, tft_df = tune_level(group_cols)
        lgbm_all.append(lgbm_df)
        tft_all.append(tft_df)

    lgbm = pd.concat(lgbm_all, ignore_index=True)
    tft = pd.concat(tft_all, ignore_index=True)
    summary = pd.concat([lgbm.groupby("level").head(5), tft.groupby("level").head(7)], ignore_index=True)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lgbm.to_csv(REPORTS_DIR / "weekly_lgbm_tuning_all.csv", index=False)
    tft.to_csv(REPORTS_DIR / "weekly_tft_quantile_tuning.csv", index=False)
    summary.to_csv(REPORTS_DIR / "weekly_lgbm_tft_tuning_summary.csv", index=False)

    print("\nTop tuning summary")
    print(summary[["level", "model", "name", "wape", "rmse", "rmsse", "bias"]].to_string(index=False, formatters={
        "wape": "{:.6f}".format,
        "rmse": "{:.3f}".format,
        "rmsse": "{:.6f}".format,
        "bias": "{:.3%}".format,
    }))


if __name__ == "__main__":
    main()
