"""
Step 2: Naive baselines for M5 forecasting.
Builds naive predictions and scores them with WRMSSE to validate the scorer.

Usage:
    python -m src.baseline
"""

import time
import numpy as np
import pandas as pd

from src.metrics.wrmsse import WRMSSEEvaluator


DATA_DIR = "data"


def load_data():
    """Load all required CSV files."""
    print("Loading data...")
    t0 = time.time()
    sales = pd.read_csv(f"{DATA_DIR}/sales_train_evaluation.csv")
    calendar = pd.read_csv(f"{DATA_DIR}/calendar.csv")
    prices = pd.read_csv(f"{DATA_DIR}/sell_prices.csv")
    print(f"  Loaded in {time.time()-t0:.1f}s | sales: {sales.shape}")
    return sales, calendar, prices


def split_train_valid(sales, holdout_days=28):
    """
    Split sales into train and validation.
    Uses last `holdout_days` of training as validation ground truth.
    We use evaluation CSV (d_1 to d_1941) and hold out d_1914-d_1941.
    """
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    id_cols = [c for c in sales.columns if not c.startswith("d_")]

    # Train: d_1 to d_1913, Valid: d_1914 to d_1941
    train_day_cols = [f"d_{i}" for i in range(1, 1914)]
    valid_day_cols = [f"d_{i}" for i in range(1914, 1942)]

    train_df = sales[id_cols + train_day_cols].copy()
    valid_df = sales[id_cols + valid_day_cols].copy()

    return train_df, valid_df, train_day_cols, valid_day_cols


def naive_last_28(train_df, train_day_cols, valid_day_cols):
    """Naive baseline: repeat last 28 days of training as forecast."""
    last_28_cols = train_day_cols[-28:]
    predictions = train_df[last_28_cols].values.copy()
    return pd.DataFrame(predictions, columns=valid_day_cols)


def naive_weekly_avg(train_df, train_day_cols, valid_day_cols):
    """
    Naive baseline: for each series, compute the average sales
    for each day-of-week over the last 4 weeks, then tile that
    weekly pattern across the 28-day horizon.
    """
    # Use last 4 weeks (28 days) to get weekly pattern
    last_28 = train_df[train_day_cols[-28:]].values  # (30490, 28)
    # Reshape to (30490, 4, 7) → mean over weeks → (30490, 7)
    weekly_pattern = last_28.reshape(last_28.shape[0], 4, 7).mean(axis=1)
    # Tile to 28 days: (30490, 7) → (30490, 28)
    predictions = np.tile(weekly_pattern, (1, 4))
    return pd.DataFrame(predictions, columns=valid_day_cols)


def naive_zero(train_df, valid_day_cols):
    """Naive baseline: predict all zeros (worst case reference)."""
    predictions = np.zeros((len(train_df), 28))
    return pd.DataFrame(predictions, columns=valid_day_cols)


def naive_global_mean(train_df, train_day_cols, valid_day_cols):
    """Naive baseline: predict the overall mean of the last 28 days per series."""
    last_28 = train_df[train_day_cols[-28:]].values
    mean_val = last_28.mean(axis=1, keepdims=True)  # (30490, 1)
    predictions = np.tile(mean_val, (1, 28))
    return pd.DataFrame(predictions, columns=valid_day_cols)


def main():
    sales, calendar, prices = load_data()
    train_df, valid_df, train_day_cols, valid_day_cols = split_train_valid(sales)

    print("\nInitialising WRMSSE evaluator...")
    t0 = time.time()
    evaluator = WRMSSEEvaluator(
        train_df=train_df,
        valid_df=valid_df,
        calendar=calendar,
        prices=prices,
    )
    print(f"  Evaluator ready in {time.time()-t0:.1f}s")

    # --- Baselines ---
    baselines = {
        "Naive (repeat last 28d)": naive_last_28(train_df, train_day_cols, valid_day_cols),
        "Naive (weekly avg, last 4w)": naive_weekly_avg(train_df, train_day_cols, valid_day_cols),
        "Naive (mean of last 28d)": naive_global_mean(train_df, train_day_cols, valid_day_cols),
        "Naive (all zeros)": naive_zero(train_df, valid_day_cols),
    }

    print("\n" + "=" * 60)
    print(f"{'Baseline':<32} {'WRMSSE':>8}")
    print("=" * 60)

    for name, preds in baselines.items():
        t0 = time.time()
        score = evaluator.score(preds)
        elapsed = time.time() - t0
        print(f"{name:<32} {score:>8.4f}  ({elapsed:.1f}s)")

    # Detailed per-level breakdown for best naive
    print("\n\nPer-level breakdown (Naive repeat last 28d):")
    print("-" * 50)
    best_preds = baselines["Naive (repeat last 28d)"]
    level_scores = evaluator.score_per_level(best_preds)
    for level, score in level_scores.items():
        print(f"  {level:<12} {score:.4f}")


if __name__ == "__main__":
    main()
