import numpy as np
import pandas as pd

from trafficml.models.baselines import HistoricalAverage, persistence, time_of_week


def test_persistence_uses_last_valid_reading_and_fallback():
    x = np.array([[[1.0, np.nan], [2.0, np.nan], [np.nan, np.nan]]])  # (S=1, history=3, N=2)
    fallback = np.full((1, 2, 2), 9.0)
    pred = persistence(x, horizon=2, fallback=fallback)
    np.testing.assert_array_equal(pred[0, :, 0], [2.0, 2.0])
    np.testing.assert_array_equal(pred[0, :, 1], [9.0, 9.0])


def test_time_of_week_slots():
    idx = pd.DatetimeIndex(["2012-03-05 00:00", "2012-03-05 00:05", "2012-03-11 23:55"])
    np.testing.assert_array_equal(time_of_week(idx), [0, 1, 2015])  # 5 Mar 2012 is a Monday


def test_historical_average_slot_means_and_fallback():
    idx = pd.date_range("2012-03-05", periods=2 * 2016, freq="5min")  # two full weeks
    values = np.tile(np.arange(2016, dtype=float), 2)[:, None].repeat(2, axis=1)
    values[:, 1] = np.nan  # sensor never observed -> NaN fallback chain ends in NaN
    values[0, 1] = 50.0  # one reading -> overall mean fills every slot
    ha = HistoricalAverage().fit(values, idx)
    pred = ha.predict(np.array([0]), idx, horizon=3)
    np.testing.assert_array_equal(pred[0, :, 0], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(pred[0, :, 1], [50.0, 50.0, 50.0])
