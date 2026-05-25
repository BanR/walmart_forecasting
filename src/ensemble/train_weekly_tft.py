"""
Train a weekly aggregate TFT for M5 store-category reconciliation.

The forecast horizon is aligned to the M5 validation window:

    week_1 = d_1914..d_1920
    week_2 = d_1921..d_1927
    week_3 = d_1928..d_1934
    week_4 = d_1935..d_1941

Rows before d_1914 are grouped into 7-day blocks using the same alignment, so
the model learns weekly aggregate demand on the same horizon definition used by
src.ensemble.weekly_reconcile.

Usage:

    python -m src.ensemble.train_weekly_tft \
      --group-cols store_id,cat_id \
      --output data/tft_weekly_store_cat_predictions.csv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss


DATA_DIR = Path("data")
VALID_START_D = 1914
VALID_END_D = 1941
PREDICTION_WEEKS = 4
MAX_ENCODER_WEEKS = 104
MIN_ENCODER_WEEKS = 52
DEFAULT_QUANTILES = [0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98]


def parse_group_cols(value: str) -> list[str]:
    cols = [c.strip() for c in value.split(",") if c.strip()]
    allowed = {"state_id", "store_id", "cat_id", "dept_id", "item_id"}
    unknown = set(cols) - allowed
    if unknown:
        raise ValueError(f"Unsupported group columns: {sorted(unknown)}")
    if not cols:
        raise ValueError("At least one group column is required")
    return cols


def _state_snap(row: pd.Series) -> int:
    if row["state_id"] == "CA":
        return row["snap_CA"]
    if row["state_id"] == "TX":
        return row["snap_TX"]
    if row["state_id"] == "WI":
        return row["snap_WI"]
    return 0


def build_weekly_frame(group_cols: list[str]) -> pd.DataFrame:
    sales = pd.read_csv(DATA_DIR / "sales_train_evaluation.csv")
    calendar = pd.read_csv(DATA_DIR / "calendar.csv")
    prices = pd.read_csv(DATA_DIR / "sell_prices.csv")

    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [f"d_{i}" for i in range(1, VALID_END_D + 1)]

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
    cal = cal[cal["d_num"].between(1, VALID_END_D)]
    cal["horizon_week"] = np.floor((cal["d_num"] - VALID_START_D) / 7).astype(int)
    cal["has_event"] = cal["event_name_1"].notna().astype(int)
    cal["is_christmas"] = ((cal["date"].dt.month == 12) & (cal["date"].dt.day == 25)).astype(int)

    min_horizon_week = int(cal["horizon_week"].min())
    cal["time_idx"] = cal["horizon_week"] - min_horizon_week

    daily = daily.merge(cal, on="d", how="inner")
    daily["snap"] = daily.apply(_state_snap, axis=1).astype(np.int8)
    daily = daily.drop(columns=["snap_CA", "snap_TX", "snap_WI"])

    daily = daily.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    daily["sell_price"] = daily["sell_price"].fillna(0.0)
    daily["revenue"] = daily["sales"] * daily["sell_price"]

    weekly = daily.groupby(group_cols + ["horizon_week", "time_idx"], as_index=False).agg(
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

    # Drop the first partial aligned week. All remaining rows are 7-day totals.
    weekly["days_in_week"] = weekly["week_end_d"] - weekly["week_start_d"] + 1
    weekly = weekly[weekly["days_in_week"] == 7].copy()
    min_time_idx = int(weekly["time_idx"].min())
    weekly["time_idx"] = weekly["time_idx"] - min_time_idx
    weekly["group_id"] = weekly[group_cols].astype(str).agg("_".join, axis=1)
    weekly["month_str"] = weekly["month"].astype(str)
    weekly["year_str"] = weekly["year"].astype(str)
    weekly["sales"] = weekly["sales"].astype("float32")
    weekly["revenue"] = weekly["revenue"].astype("float32")
    weekly["sell_price_mean"] = weekly["sell_price_mean"].astype("float32")
    weekly["snap_days"] = weekly["snap_days"].astype("float32")
    weekly["event_days"] = weekly["event_days"].astype("float32")
    weekly["christmas_days"] = weekly["christmas_days"].astype("float32")
    weekly["log_sales"] = np.log1p(weekly["sales"])

    categorical_cols = ["group_id", *group_cols, "month_str", "year_str"]
    for col in categorical_cols:
        weekly[col] = weekly[col].astype(str)

    weekly = weekly.sort_values(["group_id", "time_idx"]).reset_index(drop=True)
    return weekly


def add_training_weights(weekly: pd.DataFrame, mode: str) -> pd.DataFrame:
    weekly = weekly.copy()
    first_validation_time_idx = int(weekly.loc[weekly["week_start_d"] == VALID_START_D, "time_idx"].min())
    train = weekly[weekly["time_idx"] < first_validation_time_idx]

    if mode == "none":
        weekly["sample_weight"] = 1.0
        return weekly

    if mode == "revenue":
        basis = train.groupby("group_id")["revenue"].mean()
    elif mode == "sales":
        basis = train.groupby("group_id")["sales"].mean()
    else:
        raise ValueError(f"Unsupported weight mode: {mode}")

    median = float(basis.median())
    if median <= 0:
        weekly["sample_weight"] = 1.0
        return weekly

    weights = (basis / median).clip(0.25, 4.0)
    weekly["sample_weight"] = weekly["group_id"].map(weights).fillna(1.0).astype("float32")
    return weekly


def build_datasets(
    weekly: pd.DataFrame,
    weight_mode: str,
    use_lags: bool,
) -> tuple[TimeSeriesDataSet, TimeSeriesDataSet, int]:
    weekly = weekly.copy()
    float_cols = [
        "sales",
        "log_sales",
        "sell_price_mean",
        "snap_days",
        "event_days",
        "christmas_days",
        "sample_weight",
    ]
    for col in float_cols:
        if col in weekly.columns:
            weekly[col] = weekly[col].astype("float32")

    first_validation_time_idx = int(weekly.loc[weekly["week_start_d"] == VALID_START_D, "time_idx"].min())
    training_cutoff = first_validation_time_idx - 1
    weight_col = "sample_weight" if weight_mode != "none" else None
    lags = {"sales": [1, 2, 4, 8, 52]} if use_lags else None

    training = TimeSeriesDataSet(
        weekly[weekly["time_idx"] <= training_cutoff],
        time_idx="time_idx",
        target="sales",
        group_ids=["group_id"],
        min_encoder_length=MIN_ENCODER_WEEKS,
        max_encoder_length=MAX_ENCODER_WEEKS,
        min_prediction_length=PREDICTION_WEEKS,
        max_prediction_length=PREDICTION_WEEKS,
        static_categoricals=["group_id"],
        time_varying_known_categoricals=["month_str", "year_str"],
        time_varying_known_reals=[
            "time_idx",
            "snap_days",
            "event_days",
            "christmas_days",
            "sell_price_mean",
        ],
        time_varying_unknown_reals=["sales", "log_sales"],
        target_normalizer=GroupNormalizer(groups=["group_id"], transformation="softplus"),
        weight=weight_col,
        lags=lags,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=False,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training,
        weekly,
        min_prediction_idx=first_validation_time_idx,
        predict=True,
        stop_randomization=True,
    )
    return training, validation, first_validation_time_idx


def build_model(training: TimeSeriesDataSet, learning_rate: float) -> TemporalFusionTransformer:
    return TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=learning_rate,
        hidden_size=24,
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=12,
        loss=QuantileLoss(),
        optimizer="adam",
        reduce_on_plateau_patience=4,
    )


def train_model(
    model: TemporalFusionTransformer,
    training: TimeSeriesDataSet,
    validation: TimeSeriesDataSet,
    max_epochs: int,
    batch_size: int,
    model_dir: Path,
) -> pl.Trainer:
    train_dl = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dl = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(
        dirpath=model_dir,
        filename="weekly_tft_best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    early_stop = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=8, mode="min")
    logger = CSVLogger("lightning_logs", name="tft_weekly_store_cat")

    accelerator = "mps" if torch.backends.mps.is_available() else "cpu"
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=1,
        gradient_clip_val=0.1,
        callbacks=[checkpoint, early_stop],
        logger=logger,
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=5,
    )
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
    return trainer


def predict_weekly(
    validation: TimeSeriesDataSet,
    checkpoint_path: str,
    weekly: pd.DataFrame,
    group_cols: list[str],
    output: Path,
    batch_size: int,
    point_quantile: float,
    quantile_output: Path | None,
) -> pd.DataFrame:
    model = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path)
    val_dl = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
    accelerator = "mps" if torch.backends.mps.is_available() else "cpu"
    predictions = model.predict(
        val_dl,
        mode="raw",
        return_x=True,
        trainer_kwargs={"accelerator": accelerator, "devices": 1},
    )

    pred_tensor = predictions.output.prediction
    pred = pred_tensor.detach().cpu().numpy()
    quantile_idx = int(np.argmin(np.abs(np.asarray(DEFAULT_QUANTILES) - point_quantile)))
    point = pred[:, :, quantile_idx]

    decoder_groups = predictions.x["decoder_cat"][:, 0, 0].detach().cpu().numpy()
    group_encoder = validation.get_transformer("group_id")
    group_ids = group_encoder.inverse_transform(decoder_groups)

    group_map = weekly[["group_id", *group_cols]].drop_duplicates()
    out = pd.DataFrame({"group_id": group_ids})
    for i in range(PREDICTION_WEEKS):
        out[f"week_{i + 1}"] = np.maximum(0.0, point[:, i])
    out = out.merge(group_map, on="group_id", how="left")
    out = out[[*group_cols, *[f"week_{i}" for i in range(1, PREDICTION_WEEKS + 1)]]]

    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)

    if quantile_output is not None:
        quantile_rows = pd.DataFrame({"group_id": group_ids})
        for week_idx in range(PREDICTION_WEEKS):
            for q_idx, quantile in enumerate(DEFAULT_QUANTILES):
                q_name = str(quantile).replace(".", "")
                quantile_rows[f"week_{week_idx + 1}_q{q_name}"] = np.maximum(0.0, pred[:, week_idx, q_idx])
        quantile_rows = quantile_rows.merge(group_map, on="group_id", how="left")
        quantile_rows = quantile_rows[[*group_cols, *[c for c in quantile_rows.columns if c.startswith("week_")]]]
        quantile_output.parent.mkdir(parents=True, exist_ok=True)
        quantile_rows.to_csv(quantile_output, index=False)

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-cols", default="store_id,cat_id")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "tft_weekly_store_cat_predictions.csv")
    parser.add_argument("--weekly-data", type=Path, default=DATA_DIR / "tft_weekly_store_cat_aligned.parquet")
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--weight-mode", choices=["none", "revenue", "sales"], default="none")
    parser.add_argument("--use-lags", action="store_true")
    parser.add_argument("--point-quantile", type=float, default=0.5)
    parser.add_argument("--quantile-output", type=Path, default=None)
    args = parser.parse_args()

    t0 = time.time()
    pl.seed_everything(42)
    group_cols = parse_group_cols(args.group_cols)

    if args.weekly_data.exists():
        print(f"Loading aligned weekly data: {args.weekly_data}")
        weekly = pd.read_parquet(args.weekly_data)
    else:
        print(f"Building aligned weekly data for group_cols={group_cols}")
        weekly = build_weekly_frame(group_cols)
        args.weekly_data.parent.mkdir(parents=True, exist_ok=True)
        weekly.to_parquet(args.weekly_data, index=False)
        print(f"Weekly data saved: {args.weekly_data}")
    weekly = add_training_weights(weekly, args.weight_mode)
    print(f"Rows={len(weekly):,}, groups={weekly['group_id'].nunique():,}")
    print(f"Aligned validation weeks: d_{VALID_START_D}..d_{VALID_END_D}")

    training, validation, first_validation_time_idx = build_datasets(
        weekly,
        weight_mode=args.weight_mode,
        use_lags=args.use_lags,
    )
    print(f"Training rows: {(weekly['time_idx'] < first_validation_time_idx).sum():,}")
    print(f"Validation starts at time_idx={first_validation_time_idx}")

    model = build_model(training, args.learning_rate)
    print(f"TFT parameters: {model.size() / 1e3:.1f}k")
    model_suffix = "_".join(group_cols)
    model_dir = Path(f"models/tft_weekly_{model_suffix}")
    trainer = train_model(model, training, validation, args.max_epochs, args.batch_size, model_dir)

    best_path = trainer.checkpoint_callback.best_model_path
    print(f"Best checkpoint: {best_path}")
    preds = predict_weekly(
        validation,
        best_path,
        weekly,
        group_cols,
        args.output,
        args.batch_size,
        args.point_quantile,
        args.quantile_output,
    )
    print(f"Saved weekly TFT predictions: {args.output}")
    print(preds.head().to_string(index=False))
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
