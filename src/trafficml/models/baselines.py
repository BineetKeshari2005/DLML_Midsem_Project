"""M0 persistence and M1 historical-average baselines.

Inputs use NaN for missing readings (see ``trafficml.data.masking.to_nan``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

STEPS_PER_DAY = 288
STEPS_PER_WEEK = 7 * STEPS_PER_DAY


def time_of_week(index: pd.DatetimeIndex) -> np.ndarray:
    """5-minute slot of the week, 0..2015 (Monday 00:00 = 0)."""
    return (index.day_of_week * STEPS_PER_DAY + index.hour * 12 + index.minute // 5).to_numpy()


def persistence(x: np.ndarray, horizon: int, fallback: np.ndarray | None = None) -> np.ndarray:
    """M0: repeat the last valid reading in the input window for every horizon.

    x: (S, history, N). Where the whole window is missing, ``fallback`` (S, horizon, N) is
    used if given, otherwise the prediction stays NaN.
    """
    last = pd.DataFrame(x.transpose(1, 0, 2).reshape(x.shape[1], -1)).ffill().iloc[-1]
    last = last.to_numpy().reshape(x.shape[0], x.shape[2])
    pred = np.repeat(last[:, None, :], horizon, axis=1)
    if fallback is not None:
        pred = np.where(np.isnan(pred), fallback, pred)
    return pred.astype(np.float32)


class HistoricalAverage:
    """M1: per-sensor mean speed for each time-of-week slot, fitted on training steps only.

    Slots never observed for a sensor fall back to its time-of-day mean, then its overall mean.
    """

    def fit(self, values: np.ndarray, index: pd.DatetimeIndex) -> HistoricalAverage:
        df = pd.DataFrame(values)
        tow = time_of_week(index)
        week = df.groupby(tow).mean().reindex(range(STEPS_PER_WEEK))
        day = df.groupby(tow % STEPS_PER_DAY).mean()
        day_by_week_slot = day.reindex(np.arange(STEPS_PER_WEEK) % STEPS_PER_DAY)
        week = week.fillna(day_by_week_slot.set_axis(week.index))
        self.table_ = week.fillna(df.mean()).to_numpy(dtype=np.float32)
        return self

    def predict(self, anchors: np.ndarray, index: pd.DatetimeIndex, horizon: int) -> np.ndarray:
        """(S, horizon, N) predictions for target times anchors + 1..horizon."""
        tow = time_of_week(index)
        target_steps = anchors[:, None] + np.arange(1, horizon + 1)
        return self.table_[tow[target_steps]]
