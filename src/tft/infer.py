"""
V3: TFT Inference — load best checkpoint, generate predictions, score WRMSSE.

Requires:
    - models/tft_v3/tft_best.ckpt  (from src.tft.train)
    - data/tft_training_ds.pt       (serialised TimeSeriesDataSet)
    - data/tft_data.parquet         (long-format M5 data)

Usage:
    python -m src.tft.infer

Outputs:
    data/predictions_tft_v3.parquet  — (30490 or subset) × 28 predictions
    WRMSSE score printed to stdout
"""

import warnings
warnings.filterwarnings("ignore")

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

from src.metrics.wrmsse import WRMSSEEvaluator

DATA_DIR = Path("data")
MODEL_DIR = Path("models/tft_v3")

MAX_ENCODER_LENGTH = 90
MAX_PREDICTION_LENGTH = 28
TRAIN_CUTOFF_DAY = 1913
TRAIN_START_DAY = 1549

def load_checkpoint():
    """Load best TFT checkpoint."""
    ckpt_files = sorted(MODEL_DIR.glob("*.ckpt"))
    if not ckpt_files:
        raise FileNotFoundError(
            f"No checkpoint found in {MODEL_DIR}. Run src.tft.train first."
        )
    best_ckpt = ckpt_files[0]
    print(f"Loading checkpoint: {best_ckpt}")
    model = TemporalFusionTransformer.load_from_checkpoint(str(best_ckpt))
    model.eval()
    return model

def load_validation_dataloader():
    """Reconstruct validation dataloader from saved dataset and data."""
    print("Loading saved TimeSeriesDataSet and data...")
    training_ds = torch.load(DATA_DIR / "tft_training_ds.pt", weights_only=False)
    df = pd.read_parquet(DATA_DIR / "tft_data.parquet")

    validation_ds = TimeSeriesDataSet.from_dataset(
        training_ds, df, predict=True, stop_randomization=True
    )
    val_dl = validation_ds.to_dataloader(
        train=False, batch_size=256, num_workers=0
    )
    print(f"  Validation samples: {len(validation_ds):,}")
    return val_dl, validation_ds, df

def predict(model, val_dl):
    """
    Run inference and return median predictions.
    TFT with QuantileLoss returns 7 quantiles per timestep.
    We take the median (q=0.5, index 3) as point forecast for WRMSSE.
    """
    print("\nRunning TFT inference...")
    t0 = time.time()

    trainer = pl.Trainer(accelerator="mps", devices=1, logger=False, enable_progress_bar=True)
    predictions = model.predict(
        val_dl,
        mode="raw",
        return_x=True,
        trainer_kwargs=dict(accelerator="mps", devices=1),
    )
    print(f"  Inference complete in {time.time()-t0:.1f}s")
    return predictions

def build_prediction_df(predictions, validation_ds, df, sample_stores):
    """
    Convert TFT raw predictions to (n_series × 28) wide DataFrame
    for WRMSSE scoring.

    TFT raw output shape: (n_samples, prediction_length, n_quantiles)
    We extract median (quantile index 3 of 7).
    """
    print("\nReshaping predictions to wide format...")

    # Extract median (q0.5) predictions: shape (n_samples, 28)
    pred_tensor = predictions.output.prediction  # (n_samples, 28, 7)
    median_preds = pred_tensor[:, :, 3].cpu().numpy()  # (n_samples, 28)

    # Get the series IDs and start time indices from the dataset index
    index = validation_ds.index
    # index has columns: time_idx_first_prediction, series_id (and others)
    series_ids = index["series_id"].values if "series_id" in index.columns else None

    # If series_id not directly available, reconstruct from dataset decoder_target
    if series_ids is None:
        # Fall back: use the order from the dataloader
        series_ids = df["series_id"].unique()[:len(median_preds)]

    # Map series_id → (item_id, store_id)
    series_map = df[["series_id", "item_id", "store_id"]].drop_duplicates().set_index("series_id")

    valid_day_cols = [f"d_{i}" for i in range(1914, 1942)]

    rows = []
    for i, sid in enumerate(series_ids):
        if sid not in series_map.index:
            continue
        item_id = series_map.loc[sid, "item_id"]
        store_id = series_map.loc[sid, "store_id"]
        row = {"item_id": item_id, "store_id": store_id}
        for j, col in enumerate(valid_day_cols):
            row[col] = max(0.0, float(median_preds[i, j]))
        rows.append(row)

    pred_long = pd.DataFrame(rows)
    print(f"  Prediction rows: {len(pred_long):,}")
    return pred_long, valid_day_cols

def align_to_full_sales(pred_long, valid_day_cols, sample_stores):
    """
    Align predictions to full 30,490 × 28 matrix for WRMSSE.
    Series not in pred_long (stores not in sample) get filled with 0.
    """
    sales_ids = pd.read_csv(
        DATA_DIR / "sales_train_evaluation.csv",
        usecols=["item_id", "store_id"]
    )

    pred_wide = sales_ids.merge(
        pred_long, on=["item_id", "store_id"], how="left"
    )
    pred_wide[valid_day_cols] = pred_wide[valid_day_cols].fillna(0)
    return pred_wide[valid_day_cols]

def evaluate_wrmsse(pred_df, sample_stores):
    """Score with WRMSSE and print per-level breakdown."""
    print("\nScoring with WRMSSE...")
    t0 = time.time()

    sales = pd.read_csv(DATA_DIR / "sales_train_evaluation.csv")
    calendar = pd.read_csv(DATA_DIR / "calendar.csv")
    prices = pd.read_csv(DATA_DIR / "sell_prices.csv")

    id_cols = [c for c in sales.columns if not c.startswith("d_")]
    train_df = sales[id_cols + [f"d_{i}" for i in range(1, 1914)]]
    valid_df = sales[id_cols + [f"d_{i}" for i in range(1914, 1942)]]

    evaluator = WRMSSEEvaluator(
        train_df=train_df, valid_df=valid_df,
        calendar=calendar, prices=prices
    )

    overall = evaluator.score(pred_df)
    per_level = evaluator.score_per_level(pred_df)

    print(f"  WRMSSE: {overall:.4f} (in {time.time()-t0:.1f}s)")
    print(f"\n  Per-level breakdown:")
    for key, val in per_level.items():
        if key != "WRMSSE":
            print(f"    {key}: {val:.4f}")

    print("\n" + "=" * 60)
    print(f"  TFT V3 WRMSSE:             {overall:.4f}")
    print(f"  LightGBM V2 WRMSSE:        0.5505")
    print(f"  LightGBM V1 WRMSSE:        0.6702")
    print(f"  Naive weekly avg:          0.7524")
    if sample_stores:
        print(f"\n  NOTE: TFT trained on {sample_stores} only.")
        print(f"  Non-sampled stores filled with 0 → inflates WRMSSE artificially.")
        print(f"  For fair comparison, run with sample_stores=None.")
    print("=" * 60)

    return overall, per_level

def main():
    SAMPLE_STORES = ["CA_1", "CA_2", "CA_3"]  # must match train.py

    model = load_checkpoint()
    val_dl, validation_ds, df = load_validation_dataloader()

    predictions = predict(model, val_dl)

    pred_long, valid_day_cols = build_prediction_df(
        predictions, validation_ds, df, SAMPLE_STORES
    )

    # Save subset predictions
    pred_long.to_parquet(DATA_DIR / "predictions_tft_v3_subset.parquet", index=False)
    print(f"  Subset predictions saved to data/predictions_tft_v3_subset.parquet")

    # Align to full 30,490 matrix (unfilled stores = 0)
    pred_full = align_to_full_sales(pred_long, valid_day_cols, SAMPLE_STORES)
    pred_full.to_parquet(DATA_DIR / "predictions_tft_v3.parquet", index=False)
    print(f"  Full predictions saved to data/predictions_tft_v3.parquet")

    # Score (note: non-sampled store rows are 0 → inflated WRMSSE for subset run)
    overall, per_level = evaluate_wrmsse(pred_full, SAMPLE_STORES)

if __name__ == "__main__":
    main()
