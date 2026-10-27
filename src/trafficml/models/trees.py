"""Independent-sensor tree models with explicit missing-target training rules."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


def fit_predict_sensor(
    kind,
    x_train,
    y_train,
    x_eval,
    fallback,
    params,
    seed=42,
    min_rows=100,
    importance=False,
    model_dir=None,
):
    """Return predictions and evidence for one sensor. Never inspect evaluation targets.

    RF jointly predicts all horizons and requires complete target vectors. XGBoost
    fits one direct estimator per horizon using that horizon's valid labels.
    """
    if kind not in ("rf", "xgb"):
        raise ValueError("kind must be rf or xgb")
    if y_train.ndim != 2 or fallback.shape != (len(x_eval), y_train.shape[1]):
        raise ValueError("incompatible target/fallback shapes")
    if model_dir is not None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=False)
    pred = np.asarray(fallback, dtype=np.float32).copy()
    counts, gains = [], []
    if kind == "rf":
        valid = np.isfinite(y_train).all(axis=1)
        counts = [int(valid.sum())] * y_train.shape[1]
        if valid.sum() >= min_rows:
            model = RandomForestRegressor(random_state=seed, n_jobs=2, **params)
            model.fit(x_train[valid], y_train[valid])
            pred = model.predict(x_eval).astype(np.float32)
            if model_dir is not None:
                joblib.dump(model, model_dir / "forest.joblib", compress=3)
            if importance:
                gains.append(model.feature_importances_.tolist())
    else:
        for h in range(y_train.shape[1]):
            valid = np.isfinite(y_train[:, h])
            counts.append(int(valid.sum()))
            if valid.sum() < min_rows:
                continue
            model = XGBRegressor(
                random_state=seed, n_jobs=2, tree_method="hist", verbosity=0, **params
            )
            model.fit(x_train[valid], y_train[valid, h])
            pred[:, h] = model.predict(x_eval)
            if model_dir is not None:
                model.save_model(model_dir / f"h{h + 1:02d}.ubj")
            if importance:
                gains.append(model.feature_importances_.tolist())
    return pred, {
        "valid_training_rows": counts,
        "fallback_horizons": [h + 1 for h, n in enumerate(counts) if n < min_rows],
        "feature_importance": np.mean(gains, axis=0).tolist() if gains else [],
    }
