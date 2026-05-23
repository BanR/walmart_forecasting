"""
Step 5: Train LightGBM single direct model for M5 forecasting.

- Tweedie loss (variance_power=1.1) for zero-inflated count data
- All lags >= 28 (direct forecasting safe)
- WRMSSE evaluation on d_1914–d_1941 holdout

Usage:
    python -m src.train
"""

import time
import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path

from src.features.build import get_feature_columns
from src.metrics.wrmsse import WRMSSEEvaluator

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

def load_features():
    """Load prebuilt feature parquets."""
    print("Loading feature parquets...")
    t0 = time.time()
    df_train = pd.read_parquet(DATA_DIR / "features_train.parquet")
    df_valid = pd.read_parquet(DATA_DIR / "features_valid.parquet")
    print(f"  Train: {df_train.shape}, Valid: {df_valid.shape} ({time.time()-t0:.1f}s)")
    return df_train, df_valid

def compute_sample_weights(df_train):
    """
    Revenue-based sample weights to align training with WRMSSE.
    WRMSSE weights series by their revenue contribution.
    Weight = mean(sell_price × sales) per item-store, normalised.
    """
    print("  Computing revenue-based sample weights...")
    revenue = df_train["sell_price"] * df_train["sales"]
    # Mean revenue per item-store as weight proxy
    item_store_revenue = revenue.groupby(
        [df_train["item_id"], df_train["store_id"]]
    ).transform("mean")
    # Normalise: mean weight = 1.0
    weights = item_store_revenue / item_store_revenue.mean()
    # Floor at 0.1 to avoid completely ignoring low-revenue items
    weights = weights.clip(lower=0.1)
    print(f"    Weight range: [{weights.min():.3f}, {weights.max():.3f}], mean: {weights.mean():.3f}")
    return weights.values.astype(np.float32)


def train_lightgbm(df_train, df_valid, feature_cols, version="v2"):
    """Train a single LightGBM model with Tweedie loss."""

    params = {
        "objective": "tweedie",
        "tweedie_variance_power": 1.1,
        "metric": "tweedie",
        "learning_rate": 0.01,
        "num_leaves": 255,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "max_bin": 255,
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42,
    }
    num_boost_round = 1500

    print(f"\nTraining LightGBM {version} (Tweedie p=1.1, lr={params['learning_rate']})")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Train rows: {len(df_train):,}")
    print(f"  Valid rows: {len(df_valid):,}")
    print(f"  Rounds: {num_boost_round} (no early stopping)")

    # Drop NaN rows in training (lags might be NaN for early days)
    train_mask = df_train[feature_cols].notna().all(axis=1)
    print(f"  Rows with all features available: {train_mask.sum():,} / {len(df_train):,}")

    X_train = df_train.loc[train_mask, feature_cols]
    y_train = df_train.loc[train_mask, "sales"]

    # Revenue-based sample weights
    sample_weights = compute_sample_weights(df_train.loc[train_mask])

    X_valid = df_valid[feature_cols]
    y_valid = df_valid["sales"]

    # Fill NaN in validation (some lags may be unavailable for edge days)
    X_valid = X_valid.fillna(0)

    dtrain = lgb.Dataset(X_train, label=y_train, weight=sample_weights, free_raw_data=False)
    dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain, free_raw_data=False)

    print(f"\n  Starting training...")
    t0 = time.time()

    callbacks = [
        lgb.log_evaluation(period=200),
    ]

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dvalid],
        valid_names=["valid"],
        callbacks=callbacks,
    )

    print(f"  Training complete in {time.time()-t0:.1f}s")
    print(f"  Final iteration: {model.current_iteration()}")

    # Save model
    model_path = MODEL_DIR / f"lgb_tweedie_{version}.txt"
    model.save_model(str(model_path))
    print(f"  Model saved to {model_path}")

    return model

def predict_and_reshape(model, df_valid, feature_cols):
    """
    Predict on validation set and reshape to (30490, 28) for WRMSSE scoring.
    The validation set is ordered by (item-store, day) from the melt operation.
    """
    X_valid = df_valid[feature_cols].fillna(0)
    preds = model.predict(X_valid)

    # Clip negatives (Tweedie can predict small negatives near zero)
    preds = np.maximum(preds, 0)

    # Reshape: df_valid has 30490 series × 28 days
    # The melt order is: all series for day 1914, then all for 1915, etc.
    # Actually need to check the order — let's pivot properly
    df_pred = df_valid[["item_id", "store_id", "d"]].copy()
    df_pred["pred"] = preds

    # Pivot to wide: rows = (item_id, store_id), columns = d_1914..d_1941
    pred_wide = df_pred.pivot_table(
        index=["item_id", "store_id"],
        columns="d",
        values="pred",
        aggfunc="first"
    )

    # Sort columns by day number
    day_cols = sorted(pred_wide.columns, key=lambda x: int(x.split("_")[1]))
    pred_wide = pred_wide[day_cols]

    # Align with sales_train_evaluation row order
    sales = pd.read_csv(DATA_DIR / "sales_train_evaluation.csv",
                        usecols=["item_id", "store_id"])
    pred_aligned = sales.merge(
        pred_wide.reset_index(), on=["item_id", "store_id"], how="left"
    )
    pred_values = pred_aligned[day_cols].values

    # Create DataFrame with proper column names (d_1914..d_1941)
    valid_day_cols = [f"d_{i}" for i in range(1914, 1942)]
    pred_df = pd.DataFrame(pred_values, columns=valid_day_cols)

    return pred_df

def evaluate_wrmsse(pred_df):
    """Score predictions with WRMSSE."""
    print("\nScoring with WRMSSE...")
    t0 = time.time()

    sales = pd.read_csv(DATA_DIR / "sales_train_evaluation.csv")
    calendar = pd.read_csv(DATA_DIR / "calendar.csv")
    prices = pd.read_csv(DATA_DIR / "sell_prices.csv")

    id_cols = [c for c in sales.columns if not c.startswith("d_")]
    train_day_cols = [f"d_{i}" for i in range(1, 1914)]
    valid_day_cols = [f"d_{i}" for i in range(1914, 1942)]

    train_df = sales[id_cols + train_day_cols]
    valid_df = sales[id_cols + valid_day_cols]

    evaluator = WRMSSEEvaluator(
        train_df=train_df,
        valid_df=valid_df,
        calendar=calendar,
        prices=prices,
    )

    overall = evaluator.score(pred_df)
    per_level = evaluator.score_per_level(pred_df)

    print(f"  WRMSSE: {overall:.4f} (computed in {time.time()-t0:.1f}s)")
    print(f"\n  Per-level breakdown:")
    for key, val in per_level.items():
        if key != "WRMSSE":
            print(f"    {key}: {val:.4f}")

    return overall, per_level

def main():
    df_train, df_valid = load_features()
    feature_cols = get_feature_columns()

    # Verify all feature columns exist
    missing = [c for c in feature_cols if c not in df_train.columns]
    if missing:
        print(f"WARNING: Missing features in train: {missing}")
        feature_cols = [c for c in feature_cols if c in df_train.columns]

    missing_v = [c for c in feature_cols if c not in df_valid.columns]
    if missing_v:
        print(f"WARNING: Missing features in valid: {missing_v}")
        feature_cols = [c for c in feature_cols if c in df_valid.columns]

    print(f"  Using {len(feature_cols)} features: {feature_cols}")

    version = "v2"

    # Train
    model = train_lightgbm(df_train, df_valid, feature_cols, version=version)

    # Free training data
    del df_train
    gc.collect()

    # Predict & reshape
    print("\nGenerating predictions...")
    pred_df = predict_and_reshape(model, df_valid, feature_cols)
    print(f"  Predictions shape: {pred_df.shape}")
    print(f"  Mean prediction: {pred_df.values.mean():.3f}")
    print(f"  Prediction range: [{pred_df.values.min():.3f}, {pred_df.values.max():.3f}]")

    # Evaluate
    overall, per_level = evaluate_wrmsse(pred_df)

    # Summary
    print("\n" + "=" * 60)
    print(f"  LightGBM (Tweedie) {version} WRMSSE: {overall:.4f}")
    print(f"  V1 WRMSSE:                   0.6702")
    print(f"  Naive repeat baseline:       0.8377")
    print(f"  Weekly avg baseline:         0.7524")
    print(f"  Improvement vs V1:           {(0.6702 - overall)/0.6702*100:.1f}%")
    print(f"  Improvement vs weekly avg:   {(0.7524 - overall)/0.7524*100:.1f}%")
    print("=" * 60)

    # Save predictions
    pred_df.to_parquet(DATA_DIR / f"predictions_lgb_{version}.parquet", index=False)
    print(f"\n  Predictions saved to data/predictions_lgb_{version}.parquet")

if __name__ == "__main__":
    main()
