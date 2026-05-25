"""
Daily Temporal Fusion Transformer (TFT) diagnostic training for M5 forecasting.

This is not the current V3 hybrid model. V3 is the LightGBM V2 daily forecast
after weekly store-department TFT correction.

Follows pytorch-forecasting stallion tutorial pattern adapted for M5.
Trains on a 365-day window (memory-feasible) with 28-day forecast horizon.

Design decisions vs. LightGBM:
- TFT takes sequences, not flat rows. Data format: long-form with time_idx
- encoder_length=90: 90 days of history per series as context window
- prediction_length=28: matches M5 competition horizon
- Subset training: 10 stores × all items (~30K series) is too large;
  we sample a manageable subset or cap encoder/batch size tightly
- Loss: QuantileLoss (predicts 7 quantiles: 0.02,0.1,0.25,0.5,0.75,0.9,0.98)
  median (q50) used for WRMSSE evaluation

Usage:
    python -m src.tft.train

Outputs:
    models/tft_v3/          — legacy daily TFT diagnostic checkpoint
    data/tft_training_ds.pt — Serialised TimeSeriesDataSet (needed for inference)
"""

import warnings
warnings.filterwarnings("ignore")

import time
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

DATA_DIR = Path("data")
MODEL_DIR = Path("models/tft_v3")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ─── Hyperparameters ──────────────────────────────────────────────────────────
MAX_ENCODER_LENGTH = 90    # 90 days of history context
MAX_PREDICTION_LENGTH = 28  # 28-day forecast horizon (M5)
TRAIN_CUTOFF_DAY = 1913    # d_1914–d_1941 = validation window
TRAIN_START_DAY = 1549     # 365-day window: d_1549–d_1913
BATCH_SIZE = 64
MAX_EPOCHS = 30
LEARNING_RATE = 0.03
HIDDEN_SIZE = 32
ATTENTION_HEADS = 2
DROPOUT = 0.1
HIDDEN_CONTINUOUS_SIZE = 16

def load_and_prepare(sample_stores=None):
    """
    Load raw data and transform to long-format TimeSeriesDataSet input.

    M5 mapping to TFT concepts (following stallion pattern):
      group_ids           = ["series_id"]  (item_id + store_id concatenated)
      time_idx            = integer day index (0-based)
      target              = "sales"
      static_categoricals = ["item_id_enc", "store_id_enc", "cat_id_enc", "dept_id_enc", "state_id_enc"]
      time_varying_known  = ["day_of_week", "month", "year", "day_of_month",
                              "is_weekend", "snap", "has_event", "is_christmas"]
      time_varying_unknown = ["sales", "sell_price", "price_norm"]
    """
    print("Loading data...")
    t0 = time.time()
    sales = pd.read_csv(DATA_DIR / "sales_train_evaluation.csv")
    calendar = pd.read_csv(DATA_DIR / "calendar.csv")
    prices = pd.read_csv(DATA_DIR / "sell_prices.csv")
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # Optionally restrict to subset of stores to manage memory
    if sample_stores is not None:
        sales = sales[sales["store_id"].isin(sample_stores)].reset_index(drop=True)
        print(f"  Restricted to {sample_stores}: {len(sales):,} series")

    # ── 1. Melt to long format ───────────────────────────────────────────────
    print("  Melting to long format...")
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [f"d_{i}" for i in range(TRAIN_START_DAY, TRAIN_CUTOFF_DAY + 1 + MAX_PREDICTION_LENGTH)]
    # Include validation window days too (d_1914-d_1941) for TimeSeriesDataSet construction
    day_cols = [c for c in day_cols if c in sales.columns]

    df = sales[id_cols + day_cols].melt(
        id_vars=id_cols, var_name="d", value_name="sales"
    )
    df["sales"] = df["sales"].astype(np.float32)

    # ── 2. Add time index (integer, 0-based from dataset start) ─────────────
    day_to_idx = {f"d_{i}": i - TRAIN_START_DAY for i in range(TRAIN_START_DAY, TRAIN_CUTOFF_DAY + 1 + MAX_PREDICTION_LENGTH)}
    df["time_idx"] = df["d"].map(day_to_idx).astype(np.int32)

    # ── 3. Series ID (group key) ─────────────────────────────────────────────
    df["series_id"] = df["item_id"] + "_" + df["store_id"]

    # ── 4. Calendar features ─────────────────────────────────────────────────
    cal = calendar[["d", "date", "wm_yr_wk", "wday", "month", "year",
                    "event_name_1", "snap_CA", "snap_TX", "snap_WI"]].copy()
    cal["date"] = pd.to_datetime(cal["date"])
    cal["day_of_week"] = cal["wday"].astype(str)          # categorical string (TFT requirement)
    cal["month_str"] = cal["month"].astype(str)
    cal["is_weekend"] = (cal["wday"].isin([1, 2])).astype(np.int8)
    cal["is_christmas"] = ((cal["date"].dt.month == 12) & (cal["date"].dt.day == 25)).astype(np.int8)
    cal["day_of_month"] = cal["date"].dt.day.astype(np.int8)
    cal["has_event"] = cal["event_name_1"].notna().astype(np.int8)

    df = df.merge(cal[["d", "wm_yr_wk", "month_str", "year", "day_of_week",
                        "is_weekend", "snap_CA", "snap_TX", "snap_WI",
                        "is_christmas", "day_of_month", "has_event"]], on="d", how="left")

    # State-specific SNAP
    df["snap"] = 0
    df.loc[df["state_id"] == "CA", "snap"] = df.loc[df["state_id"] == "CA", "snap_CA"].values
    df.loc[df["state_id"] == "TX", "snap"] = df.loc[df["state_id"] == "TX", "snap_TX"].values
    df.loc[df["state_id"] == "WI", "snap"] = df.loc[df["state_id"] == "WI", "snap_WI"].values
    df["snap"] = df["snap"].astype(np.int8)
    df.drop(columns=["snap_CA", "snap_TX", "snap_WI"], inplace=True)

    # ── 5. Price features ────────────────────────────────────────────────────
    df = df.merge(prices, on=["item_id", "store_id", "wm_yr_wk"], how="left")
    df["sell_price"] = df["sell_price"].fillna(0).astype(np.float32)

    # Price relative to department average (promotion signal)
    dept_mean_price = df.groupby("dept_id")["sell_price"].transform("mean")
    df["price_norm"] = (df["sell_price"] / dept_mean_price.replace(0, 1)).astype(np.float32)

    # ── 6. Categorical encodings (as strings for TFT) ────────────────────────
    for col in ["item_id", "dept_id", "cat_id", "store_id", "state_id"]:
        df[col] = df[col].astype(str)

    # ── 7. Log-sales as additional unknown real (helps with scale) ───────────
    df["log_sales"] = np.log1p(df["sales"]).astype(np.float32)

    # ── 8. Clip to training + validation window ──────────────────────────────
    total_days = TRAIN_CUTOFF_DAY - TRAIN_START_DAY + 1 + MAX_PREDICTION_LENGTH
    df = df[df["time_idx"] < total_days].copy()

    print(f"  Long-format shape: {df.shape}")
    print(f"  Series: {df['series_id'].nunique():,}, Days: {df['time_idx'].nunique()}")
    return df

def build_datasets(df):
    """Build TimeSeriesDataSet for training and validation."""
    # Training cutoff in time_idx units
    training_cutoff = TRAIN_CUTOFF_DAY - TRAIN_START_DAY  # last training day index

    print(f"\nBuilding TimeSeriesDataSet (training cutoff time_idx={training_cutoff})...")

    training = TimeSeriesDataSet(
        df[df.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="sales",
        group_ids=["series_id"],
        min_encoder_length=MAX_ENCODER_LENGTH // 2,
        max_encoder_length=MAX_ENCODER_LENGTH,
        min_prediction_length=1,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        # Static features: fixed per series over time
        static_categoricals=["item_id", "dept_id", "cat_id", "store_id", "state_id"],
        static_reals=[],
        # Known future features: calendar is known ahead of time
        time_varying_known_categoricals=["day_of_week", "month_str"],
        time_varying_known_reals=[
            "time_idx",
            "is_weekend", "snap", "has_event", "is_christmas",
            "day_of_month", "sell_price", "price_norm",
        ],
        # Unknown future features: only past sales are unknown
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=["sales", "log_sales"],
        target_normalizer=GroupNormalizer(
            groups=["series_id"], transformation="softplus"
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )

    # Validation: predict last MAX_PREDICTION_LENGTH days for each series
    validation = TimeSeriesDataSet.from_dataset(
        training, df, predict=True, stop_randomization=True
    )

    print(f"  Training samples: {len(training):,}")
    print(f"  Validation samples: {len(validation):,}")
    return training, validation

def build_model(training):
    """Instantiate TFT from training dataset."""
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=LEARNING_RATE,
        hidden_size=HIDDEN_SIZE,
        attention_head_size=ATTENTION_HEADS,
        dropout=DROPOUT,
        hidden_continuous_size=HIDDEN_CONTINUOUS_SIZE,
        loss=QuantileLoss(),
        log_interval=20,
        optimizer="adam",
        reduce_on_plateau_patience=4,
    )
    print(f"\nTFT parameters: {tft.size() / 1e3:.1f}k")
    return tft

def train(tft, train_dl, val_dl):
    """Train with PyTorch Lightning."""
    early_stop = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=8, mode="min", verbose=True
    )
    lr_logger = LearningRateMonitor()
    checkpoint = ModelCheckpoint(
        dirpath=str(MODEL_DIR),
        filename="tft_best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    logger = CSVLogger("lightning_logs", name="tft_daily_diagnostic")

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="mps",
        devices=1,
        gradient_clip_val=0.1,
        limit_train_batches=500,   # ~10 min/epoch; covers full dataset diversity
        callbacks=[early_stop, lr_logger, checkpoint],
        logger=logger,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    print(f"\nTraining TFT (max_epochs={MAX_EPOCHS}, limit_train_batches=500, MPS accelerated)...")
    t0 = time.time()
    trainer.fit(tft, train_dataloaders=train_dl, val_dataloaders=val_dl)
    print(f"  Training complete in {time.time()-t0:.1f}s")
    print(f"  Best checkpoint: {trainer.checkpoint_callback.best_model_path}")
    return trainer

def main():
    # Restrict to 3 stores for a manageable daily TFT diagnostic.
    # Full 10-store run: set sample_stores=None (needs ~16GB RAM)
    SAMPLE_STORES = ["CA_1", "CA_2", "CA_3"]
    print(f"Running daily TFT diagnostic on stores: {SAMPLE_STORES}")
    print("  (Set SAMPLE_STORES=None in main() for full 30K-series run)\n")

    df = load_and_prepare(sample_stores=SAMPLE_STORES)

    training_ds, validation_ds = build_datasets(df)

    # Save dataset for inference script
    torch.save(training_ds, DATA_DIR / "tft_training_ds.pt")
    print(f"  TimeSeriesDataSet saved to data/tft_training_ds.pt")

    # Also save the full df needed for inference encoder window
    df.to_parquet(DATA_DIR / "tft_data.parquet", index=False)
    print(f"  Full long-format data saved to data/tft_data.parquet")

    train_dl = training_ds.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=0)
    val_dl = validation_ds.to_dataloader(train=False, batch_size=BATCH_SIZE * 4, num_workers=0)

    tft = build_model(training_ds)
    trainer = train(tft, train_dl, val_dl)

    # Quick validation loss summary
    print(f"\nBest val_loss: {trainer.checkpoint_callback.best_model_score:.4f}")
    print(f"Model saved to: {trainer.checkpoint_callback.best_model_path}")
    print("\nRun src.tft.infer to generate predictions and compute WRMSSE.")

if __name__ == "__main__":
    main()
