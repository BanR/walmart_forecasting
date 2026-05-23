"""
WRMSSE (Weighted Root Mean Squared Scaled Error) scorer for M5 competition.

Usage:
    from src.metrics.wrmsse import WRMSSEEvaluator

    evaluator = WRMSSEEvaluator(
        train_df=train_df,       # sales_train with id columns + d_1...d_N
        valid_df=valid_df,       # actual sales for the 28-day validation window
        calendar=calendar_df,
        prices=prices_df,
    )
    score = evaluator.score(predictions)  # predictions: (30490, 28) array or DataFrame
"""

import numpy as np
import pandas as pd


# 12 hierarchy group definitions
GROUP_IDS = (
    "all_id",           # Level 1: Total
    "state_id",         # Level 2: State
    "store_id",         # Level 3: Store
    "cat_id",           # Level 4: Category
    "dept_id",          # Level 5: Department
    ["state_id", "cat_id"],     # Level 6
    ["state_id", "dept_id"],    # Level 7
    ["store_id", "cat_id"],     # Level 8
    ["store_id", "dept_id"],    # Level 9
    "item_id",                  # Level 10
    ["item_id", "state_id"],    # Level 11
    ["item_id", "store_id"],    # Level 12 (bottom)
)


class WRMSSEEvaluator:
    """Compute WRMSSE matching the M5 Kaggle competition metric."""

    def __init__(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        calendar: pd.DataFrame,
        prices: pd.DataFrame,
    ):
        # Identify day columns in training data
        train_day_cols = [c for c in train_df.columns if c.startswith("d_")]
        self.train_day_cols = train_day_cols

        # The last 28 days of training are used for revenue weights
        self.weight_day_cols = train_day_cols[-28:]

        # Add all_id column for Level 1 aggregation
        train_df = train_df.copy()
        train_df["all_id"] = "all"

        valid_df = valid_df.copy()
        if "all_id" not in valid_df.columns:
            valid_df["all_id"] = "all"

        # Identify id (non-day) columns
        self.id_columns = [c for c in train_df.columns if not c.startswith("d_")]

        # Valid day columns
        if isinstance(valid_df, pd.DataFrame):
            valid_day_cols = [c for c in valid_df.columns if c.startswith("d_")]
            if valid_day_cols:
                self.valid_day_cols = valid_day_cols
            else:
                # valid_df has no d_ columns — assume it's just the 28 value columns
                self.valid_day_cols = [f"d_{i}" for i in range(1, valid_df.shape[1] + 1)]
                valid_df.columns = self.valid_day_cols
        else:
            self.valid_day_cols = [f"F{i}" for i in range(1, 29)]

        # Attach id columns to valid_df if missing
        if "item_id" not in valid_df.columns:
            valid_df = pd.concat(
                [train_df[self.id_columns].reset_index(drop=True), valid_df.reset_index(drop=True)],
                axis=1,
            )

        self.train_df = train_df
        self.valid_df = valid_df
        self.calendar = calendar
        self.prices = prices
        self.n_series = len(train_df)

        # Precompute per-level: scales, weights, and ground truth
        self._precompute(train_df, valid_df)

    def _precompute(self, train_df, valid_df):
        """Precompute scale (denominator) and weights for all 12 levels."""
        # Revenue weight DataFrame
        weight_df = self._compute_weight_df(train_df)

        self.level_data = []

        for i, group_id in enumerate(GROUP_IDS):
            level_num = i + 1

            # Aggregate training series by group
            train_agg = train_df.groupby(group_id)[self.train_day_cols].sum()

            # Compute RMSSE denominator (scale) for each aggregated series
            scale = self._compute_scale(train_agg)

            # Aggregate validation ground truth
            valid_agg = valid_df.groupby(group_id)[self.valid_day_cols].sum()

            # Compute revenue weights for this level
            weight_agg = weight_df.groupby(group_id)[self.weight_day_cols].sum().sum(axis=1)
            weights = weight_agg / weight_agg.sum()

            self.level_data.append({
                "level": level_num,
                "group_id": group_id,
                "scale": scale,
                "valid_agg": valid_agg,
                "weights": weights,
            })

    def _compute_scale(self, train_agg: pd.DataFrame) -> np.ndarray:
        """
        Compute the RMSSE denominator for each aggregated series.
        Scale = mean of squared differences of the naive forecast,
        starting from the first non-zero observation.
        """
        data = train_agg.values  # (n_series_at_level, n_train_days)
        scales = np.zeros(data.shape[0])

        for idx in range(data.shape[0]):
            series = data[idx]
            # Find first non-zero position
            non_zero_mask = series != 0
            if not non_zero_mask.any():
                scales[idx] = 1.0  # avoid division by zero for all-zero series
                continue
            start = np.argmax(non_zero_mask)
            active_series = series[start:]
            if len(active_series) < 2:
                scales[idx] = 1.0
                continue
            # Mean squared difference (naive one-step forecast error)
            diffs = active_series[1:] - active_series[:-1]
            scales[idx] = np.mean(diffs ** 2)

        # Replace zeros with 1.0 to avoid division by zero
        scales[scales == 0] = 1.0
        return scales

    def _compute_weight_df(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Compute dollar-revenue for the weight period (last 28 training days)."""
        day_to_week = self.calendar.set_index("d")["wm_yr_wk"].to_dict()

        # Extract weight columns + identifiers
        weight_data = train_df[["item_id", "store_id"] + self.weight_day_cols].copy()
        weight_long = weight_data.melt(
            id_vars=["item_id", "store_id"], var_name="d", value_name="units"
        )
        weight_long["wm_yr_wk"] = weight_long["d"].map(day_to_week)

        # Merge with prices
        weight_long = weight_long.merge(
            self.prices, how="left", on=["item_id", "store_id", "wm_yr_wk"]
        )
        weight_long["revenue"] = weight_long["units"] * weight_long["sell_price"].fillna(0)

        # Pivot back to wide format
        weight_wide = weight_long.pivot_table(
            index=["item_id", "store_id"], columns="d", values="revenue", aggfunc="sum"
        ).fillna(0)

        # Reindex to match train_df order
        weight_wide = weight_wide.loc[
            list(zip(train_df["item_id"], train_df["store_id"]))
        ].reset_index(drop=True)

        # Attach id columns
        result = pd.concat(
            [train_df[self.id_columns].reset_index(drop=True), weight_wide.reset_index(drop=True)],
            axis=1,
        )
        return result

    def rmsse(self, predictions_agg: pd.DataFrame, level_idx: int) -> np.ndarray:
        """Compute RMSSE for each series at a given level."""
        ld = self.level_data[level_idx]
        valid_y = ld["valid_agg"].values
        pred_y = predictions_agg.values

        # MSE over the 28-day horizon for each series
        mse = np.mean((valid_y - pred_y) ** 2, axis=1)

        # RMSSE = sqrt(MSE / scale)
        rmsse_values = np.sqrt(mse / ld["scale"])
        return rmsse_values

    def score(self, predictions) -> float:
        """
        Compute WRMSSE.

        Parameters
        ----------
        predictions : np.ndarray or pd.DataFrame of shape (30490, 28)
            Bottom-level predictions for the 28-day forecast horizon.

        Returns
        -------
        float : WRMSSE score
        """
        if isinstance(predictions, np.ndarray):
            predictions = pd.DataFrame(predictions, columns=self.valid_day_cols)

        # Attach id columns to predictions
        pred_with_ids = pd.concat(
            [self.valid_df[self.id_columns].reset_index(drop=True), predictions.reset_index(drop=True)],
            axis=1,
        )

        level_scores = []
        for i, group_id in enumerate(GROUP_IDS):
            ld = self.level_data[i]

            # Aggregate predictions to this level
            pred_agg = pred_with_ids.groupby(group_id)[self.valid_day_cols].sum()

            # Reindex to match ground truth order
            pred_agg = pred_agg.reindex(ld["valid_agg"].index)

            # Compute RMSSE per series at this level
            rmsse_values = self.rmsse(pred_agg, i)

            # Weighted sum
            weights = ld["weights"].values
            weighted_rmsse = np.sum(weights * rmsse_values)
            level_scores.append(weighted_rmsse)

        # Final score: equal-weight average across 12 levels
        return np.mean(level_scores)

    def score_per_level(self, predictions) -> dict:
        """Return WRMSSE contribution per level (for diagnostics)."""
        if isinstance(predictions, np.ndarray):
            predictions = pd.DataFrame(predictions, columns=self.valid_day_cols)

        pred_with_ids = pd.concat(
            [self.valid_df[self.id_columns].reset_index(drop=True), predictions.reset_index(drop=True)],
            axis=1,
        )

        results = {}
        for i, group_id in enumerate(GROUP_IDS):
            ld = self.level_data[i]
            pred_agg = pred_with_ids.groupby(group_id)[self.valid_day_cols].sum()
            pred_agg = pred_agg.reindex(ld["valid_agg"].index)
            rmsse_values = self.rmsse(pred_agg, i)
            weights = ld["weights"].values
            weighted_rmsse = np.sum(weights * rmsse_values)
            results[f"Level_{i+1}"] = weighted_rmsse

        results["WRMSSE"] = np.mean(list(results.values()))
        return results
