"""
Step 4: Feature engineering for M5 forecasting.

Transforms the wide-format sales data into a long-format feature DataFrame
suitable for LightGBM training. Uses only lag >= 28 features (direct approach).

Usage:
    python -m src.features.build

Output:
    data/features_train.parquet  — training features
    data/features_valid.parquet  — validation features (d_1914-d_1941)
"""

import gc
import time
import numpy as np
import pandas as pd
from pathlib import Path


DATA_DIR = Path("data")


def load_data():
    """Load raw data files."""
    print("Loading data...")
    t0 = time.time()
    sales = pd.read_csv(DATA_DIR / "sales_train_evaluation.csv")
    calendar = pd.read_csv(DATA_DIR / "calendar.csv")
    prices = pd.read_csv(DATA_DIR / "sell_prices.csv")
    print(f"  Loaded in {time.time()-t0:.1f}s")
    return sales, calendar, prices


def melt_sales(sales, day_range):
    """
    Melt wide-format sales into long format for a range of days.
    Each row = (item_id, store_id, day, sales)
    """
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [f"d_{d}" for d in day_range]

    # Keep only relevant columns to save memory
    df = sales[id_cols + day_cols].copy()
    df = df.melt(id_vars=id_cols, var_name="d", value_name="sales")
    return df


def add_calendar_features(df, calendar):
    """Merge calendar data and extract temporal features."""
    cal_cols = [
        "d", "date", "wm_yr_wk", "weekday", "wday", "month", "year",
        "event_name_1", "event_type_1", "snap_CA", "snap_TX", "snap_WI",
    ]
    cal = calendar[cal_cols].copy()

    df = df.merge(cal, on="d", how="left")

    # Extract useful features
    df["day_of_week"] = df["wday"]  # 1=Saturday in Walmart encoding
    df["is_weekend"] = df["day_of_week"].isin([1, 2]).astype(np.int8)  # Sat, Sun

    # SNAP: use state-specific flag
    df["snap"] = 0
    df.loc[df.state_id == "CA", "snap"] = df.loc[df.state_id == "CA", "snap_CA"]
    df.loc[df.state_id == "TX", "snap"] = df.loc[df.state_id == "TX", "snap_TX"]
    df.loc[df.state_id == "WI", "snap"] = df.loc[df.state_id == "WI", "snap_WI"]
    df["snap"] = df["snap"].astype(np.int8)
    df.drop(columns=["snap_CA", "snap_TX", "snap_WI"], inplace=True)

    # Event encoding
    df["has_event"] = df["event_name_1"].notna().astype(np.int8)
    df["event_type_enc"] = df["event_type_1"].map(
        {"Sporting": 1, "Cultural": 2, "National": 3, "Religious": 4}
    ).fillna(0).astype(np.int8)
    df.drop(columns=["event_name_1", "event_type_1"], inplace=True)

    return df


def add_price_features(df, prices):
    """Merge sell prices and compute price-related features."""
    df = df.merge(prices, on=["item_id", "store_id", "wm_yr_wk"], how="left")

    # Price momentum features (computed later via rolling on the series)
    # For now just include raw price
    df["sell_price"] = df["sell_price"].fillna(0).astype(np.float32)

    return df


def add_lag_features(df, sales, lag_days):
    """
    Add lag features using the original wide-format sales data.
    Only uses lags >= 28 (safe for 28-day direct forecasting).
    """
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    day_to_idx = {d: i for i, d in enumerate(day_cols)}
    n_days = len(day_cols)

    # Build a lookup: (item_id, store_id) -> row index in sales
    id_cols = ["item_id", "store_id"]
    sales_indexed = sales.set_index(id_cols)
    sales_values = sales[day_cols].values  # (30490, n_days)

    # Map each item-store to its row index
    item_store_to_row = {
        (row.item_id, row.store_id): idx
        for idx, row in sales[id_cols].iterrows()
    }

    # Get day indices for the df
    df["d_idx"] = df["d"].map(day_to_idx)

    # For each lag, vectorised lookup
    for lag in lag_days:
        print(f"    lag_{lag}...")
        source_idx = df["d_idx"].values - lag
        # Map item-store pairs to row indices
        row_indices = df[["item_id", "store_id"]].apply(
            lambda r: item_store_to_row.get((r.item_id, r.store_id), -1), axis=1
        ).values

        # Vectorised: only valid where source_idx >= 0
        valid_mask = (source_idx >= 0) & (source_idx < n_days) & (row_indices >= 0)
        lag_values = np.full(len(df), np.nan, dtype=np.float32)
        lag_values[valid_mask] = sales_values[
            row_indices[valid_mask], source_idx[valid_mask]
        ]
        df[f"lag_{lag}"] = lag_values

    df.drop(columns=["d_idx"], inplace=True)
    return df


def add_lag_features_fast(df, sales):
    """
    Fast vectorised lag feature computation.
    Only computes lags >= 28 (safe for direct 28-step forecasting).
    """
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    day_to_idx = {d: i for i, d in enumerate(day_cols)}
    sales_values = sales[day_cols].values.astype(np.float32)  # (30490, n_days)

    # Map item_id + store_id → row in sales array
    id_pairs = list(zip(sales["item_id"], sales["store_id"]))
    pair_to_row = {pair: i for i, pair in enumerate(id_pairs)}

    # Get row and day indices for each record in df
    df_pairs = list(zip(df["item_id"], df["store_id"]))
    row_indices = np.array([pair_to_row[p] for p in df_pairs], dtype=np.int32)
    day_indices = df["d"].map(day_to_idx).values.astype(np.int32)

    n_days = len(day_cols)
    lag_days = [28, 29, 30, 35, 42, 49, 56, 91, 182, 364]
    for lag in lag_days:
        print(f"    lag_{lag}...")
        src_idx = day_indices - lag
        valid = (src_idx >= 0) & (src_idx < n_days)
        vals = np.full(len(df), np.nan, dtype=np.float32)
        vals[valid] = sales_values[row_indices[valid], src_idx[valid]]
        df[f"lag_{lag}"] = vals

    return df


def add_rolling_features_fast(df, sales):
    """
    Compute rolling statistics using the original wide-format data.
    All windows end at lag-28 or earlier (safe for direct forecasting).
    """
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    day_to_idx = {d: i for i, d in enumerate(day_cols)}
    sales_values = sales[day_cols].values.astype(np.float32)
    n_days = len(day_cols)

    # Map item-store to row
    id_pairs = list(zip(sales["item_id"], sales["store_id"]))
    pair_to_row = {pair: i for i, pair in enumerate(id_pairs)}

    df_pairs = list(zip(df["item_id"], df["store_id"]))
    row_indices = np.array([pair_to_row[p] for p in df_pairs], dtype=np.int32)
    day_indices = df["d"].map(day_to_idx).values.astype(np.int32)

    # Rolling windows: compute mean/std of the window ending at (day - 28)
    # i.e., for predicting day t, we use data up to t-28
    windows = [7, 14, 28, 56, 180]

    for window in windows:
        print(f"    rolling_mean_{window} & rolling_std_{window}...")
        # End index: day_idx - 28 (inclusive)
        # Start index: day_idx - 28 - window + 1
        end_idx = day_indices - 28  # last known day
        start_idx = end_idx - window + 1

        means = np.full(len(df), np.nan, dtype=np.float32)
        stds = np.full(len(df), np.nan, dtype=np.float32)

        valid = (start_idx >= 0) & (end_idx < n_days) & (end_idx >= 0)

        # Process in chunks to manage memory
        chunk_size = 100000
        for chunk_start in range(0, len(df), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(df))
            chunk_valid = valid[chunk_start:chunk_end]
            if not chunk_valid.any():
                continue

            chunk_rows = row_indices[chunk_start:chunk_end][chunk_valid]
            chunk_starts = start_idx[chunk_start:chunk_end][chunk_valid]
            chunk_ends = end_idx[chunk_start:chunk_end][chunk_valid]

            for j, (r, s, e) in enumerate(zip(chunk_rows, chunk_starts, chunk_ends)):
                window_data = sales_values[r, s:e+1]
                idx = chunk_start + np.where(chunk_valid)[0][j]
                means[idx] = window_data.mean()
                stds[idx] = window_data.std()

        df[f"rolling_mean_{window}"] = means
        df[f"rolling_std_{window}"] = stds

    return df


def add_rolling_features_vectorised(df, sales):
    """
    Precompute cumulative sums for fast rolling statistics.
    All windows end at lag-28 (safe for direct 28-step forecasting).
    """
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    day_to_idx = {d: i for i, d in enumerate(day_cols)}
    sales_values = sales[day_cols].values.astype(np.float32)
    n_days = len(day_cols)
    n_series = sales_values.shape[0]

    # Precompute cumulative sum and cumulative sum of squares for each series
    print("    Precomputing cumulative sums...")
    cumsum = np.zeros((n_series, n_days + 1), dtype=np.float64)
    cumsum[:, 1:] = np.cumsum(sales_values, axis=1)

    cumsum_sq = np.zeros((n_series, n_days + 1), dtype=np.float64)
    cumsum_sq[:, 1:] = np.cumsum(sales_values ** 2, axis=1)

    # Map rows
    id_pairs = list(zip(sales["item_id"], sales["store_id"]))
    pair_to_row = {pair: i for i, pair in enumerate(id_pairs)}
    df_pairs = list(zip(df["item_id"], df["store_id"]))
    row_indices = np.array([pair_to_row[p] for p in df_pairs], dtype=np.int32)
    day_indices = df["d"].map(day_to_idx).values.astype(np.int32)

    windows = [7, 14, 28, 56, 180]
    for window in windows:
        print(f"    rolling_mean_{window} / rolling_std_{window}...")
        end_idx = day_indices - 28  # last fully known day index
        start_idx = end_idx - window + 1

        valid = (start_idx >= 0) & (end_idx < n_days) & (end_idx >= 0)

        # Compute rolling mean: (cumsum[end+1] - cumsum[start]) / window
        means = np.full(len(df), np.nan, dtype=np.float32)
        stds = np.full(len(df), np.nan, dtype=np.float32)

        v_rows = row_indices[valid]
        v_starts = start_idx[valid]
        v_ends = end_idx[valid]

        sum_window = cumsum[v_rows, v_ends + 1] - cumsum[v_rows, v_starts]
        sum_sq_window = cumsum_sq[v_rows, v_ends + 1] - cumsum_sq[v_rows, v_starts]

        m = sum_window / window
        var = sum_sq_window / window - m ** 2
        var = np.maximum(var, 0)  # numerical stability

        means[valid] = m.astype(np.float32)
        stds[valid] = np.sqrt(var).astype(np.float32)

        df[f"rolling_mean_{window}"] = means
        df[f"rolling_std_{window}"] = stds

    del cumsum, cumsum_sq
    gc.collect()
    return df


def add_categorical_encodings(df):
    """Label-encode categorical features for LightGBM."""
    cat_cols = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
    for col in cat_cols:
        df[col + "_enc"] = df[col].astype("category").cat.codes.astype(np.int16)
    return df


def add_price_momentum(df):
    """Price change features relative to rolling price."""
    # Price relative features (simple version — grouped computation)
    # For large datasets, this approximation works
    df["price_norm"] = df.groupby("dept_id")["sell_price"].transform(
        lambda x: x / x.mean() if x.mean() > 0 else 0
    ).astype(np.float32)
    return df


def add_eda_driven_features(df, sales, calendar):
    """
    Features identified from EDA deep-dive:
    - snap_x_foods: SNAP × FOODS interaction (WI FOODS lift = +27.8%)
    - is_christmas: Store closure day
    - rolling_zero_ratio: How intermittent is this series?
    - price_change_from_lag28: Promotional signal
    - year: Trend proxy (all depts growing 30-115%)
    - day_of_month: SNAP issuance timing
    """
    print("  Adding EDA-driven features...")

    # 1. SNAP × FOODS interaction (biggest SNAP effect is on FOODS category)
    foods_mask = df["cat_id"] == "FOODS"
    df["snap_x_foods"] = (df["snap"] * foods_mask.astype(np.int8)).astype(np.int8)

    # 2. Christmas flag (stores close — predict 0)
    cal_date = calendar[["d", "date"]].copy()
    cal_date["date"] = pd.to_datetime(cal_date["date"])
    cal_date["is_christmas"] = ((cal_date["date"].dt.month == 12) &
                                 (cal_date["date"].dt.day == 25)).astype(np.int8)
    cal_date["day_of_month"] = cal_date["date"].dt.day.astype(np.int8)
    df = df.merge(cal_date[["d", "is_christmas", "day_of_month"]], on="d", how="left")

    # 3. Rolling zero ratio (from rolling_mean_28: if mean ≈ 0, item rarely sells)
    if "rolling_mean_28" in df.columns:
        df["rolling_zero_ratio"] = (df["rolling_mean_28"] <= 0.05).astype(np.int8)
    else:
        df["rolling_zero_ratio"] = np.int8(0)

    # 4. Price change from lag_28 (promotional signal)
    if "lag_28" in df.columns:
        # Use sell_price / rolling_mean_price as proxy (price relative to recent)
        # A simpler version: if price is below department norm, it's a promotion
        df["price_below_norm"] = (df["price_norm"] < 0.95).astype(np.int8)
    else:
        df["price_below_norm"] = np.int8(0)

    # 5. Year (trend proxy — captures organic growth)
    if "year" not in df.columns:
        year_map = calendar.set_index("d")["year"].to_dict()
        df["year"] = df["d"].map(year_map).astype(np.int16)

    return df


def build_features(
    sales, calendar, prices,
    train_days=range(1, 1914),
    valid_days=range(1914, 1942),
    use_recent_only=True,
    recent_n_days=365,
):
    """
    Build the full feature DataFrame.

    Parameters
    ----------
    train_days : range of day numbers for training rows
    valid_days : range of day numbers for validation target rows
    use_recent_only : if True, only use the last `recent_n_days` for training
                      (avoids 55M+ row memory issues)
    """
    # For training: use recent history to keep manageable
    if use_recent_only:
        actual_train_days = range(max(train_days.start, train_days.stop - recent_n_days), train_days.stop)
    else:
        actual_train_days = train_days

    print(f"\nBuilding features for training: d_{actual_train_days.start}–d_{actual_train_days.stop-1}")
    print(f"  Validation: d_{valid_days.start}–d_{valid_days.stop-1}")
    print(f"  Training rows: ~{len(actual_train_days) * 30490:,}")

    # Melt to long format
    print("\n  Melting sales to long format (train)...")
    t0 = time.time()
    df_train = melt_sales(sales, actual_train_days)
    print(f"    → {len(df_train):,} rows in {time.time()-t0:.1f}s")

    print("  Melting sales to long format (valid)...")
    df_valid = melt_sales(sales, valid_days)
    print(f"    → {len(df_valid):,} rows")

    # Combine for feature computation (then split later)
    df = pd.concat([df_train, df_valid], ignore_index=True)
    df["is_train"] = np.concatenate([
        np.ones(len(df_train), dtype=np.int8),
        np.zeros(len(df_valid), dtype=np.int8)
    ])
    del df_train, df_valid
    gc.collect()

    # Add features
    print("\n  Adding calendar features...")
    df = add_calendar_features(df, calendar)

    print("  Adding price features...")
    df = add_price_features(df, prices)

    print("  Adding lag features (>= 28 days, direct forecasting safe)...")
    df = add_lag_features_fast(df, sales)

    print("  Adding rolling features (vectorised)...")
    df = add_rolling_features_vectorised(df, sales)

    print("  Adding categorical encodings...")
    df = add_categorical_encodings(df)

    print("  Adding price momentum...")
    df = add_price_momentum(df)

    df = add_eda_driven_features(df, sales, calendar)

    # Split back
    df_train = df[df.is_train == 1].drop(columns=["is_train"]).reset_index(drop=True)
    df_valid = df[df.is_train == 0].drop(columns=["is_train"]).reset_index(drop=True)
    del df
    gc.collect()

    return df_train, df_valid


def get_feature_columns():
    """Return list of feature column names for modelling."""
    features = [
        # Calendar
        "day_of_week", "month", "year", "day_of_month",
        "is_weekend", "snap", "has_event", "event_type_enc",
        "is_christmas",
        # Price
        "sell_price", "price_norm", "price_below_norm",
        # EDA-driven interactions
        "snap_x_foods", "rolling_zero_ratio",
        # Lags
        "lag_28", "lag_29", "lag_30", "lag_35", "lag_42", "lag_49",
        "lag_56", "lag_91", "lag_182", "lag_364",
        # Rolling
        "rolling_mean_7", "rolling_std_7",
        "rolling_mean_14", "rolling_std_14",
        "rolling_mean_28", "rolling_std_28",
        "rolling_mean_56", "rolling_std_56",
        "rolling_mean_180", "rolling_std_180",
        # Categorical encodings
        "item_id_enc", "dept_id_enc", "cat_id_enc", "store_id_enc", "state_id_enc",
    ]
    return features


def main():
    sales, calendar, prices = load_data()

    df_train, df_valid = build_features(
        sales, calendar, prices,
        train_days=range(1, 1914),
        valid_days=range(1914, 1942),
        use_recent_only=True,
        recent_n_days=730,
    )

    print(f"\n  Train shape: {df_train.shape}")
    print(f"  Valid shape: {df_valid.shape}")
    print(f"  Feature columns: {get_feature_columns()}")

    # Save to parquet
    print("\n  Saving to parquet...")
    df_train.to_parquet(DATA_DIR / "features_train.parquet", index=False)
    df_valid.to_parquet(DATA_DIR / "features_valid.parquet", index=False)
    print("  Done! Files saved to data/features_train.parquet and data/features_valid.parquet")


if __name__ == "__main__":
    main()
