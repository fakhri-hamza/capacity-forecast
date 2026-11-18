import numpy as np
import pandas as pd

from capfc.detect import alarms, evaluate, zscore_detector
from capfc.forecast import mase


def series(values):
    return pd.Series(values, index=pd.date_range("2024-01-01", periods=len(values), freq="5min"))


def test_spike_after_probation_is_detected_and_counted():
    rng = np.random.default_rng(0)
    v = 50 + rng.normal(0, 1, 2000)
    v[1500] = 90                                   # one obvious spike
    s = series(v)
    al = alarms(zscore_detector(s))
    window = [(s.index[1495], s.index[1505])]
    r = evaluate(al, window)
    assert r["detected"] == 1
    # pure noise: over 1,700 points a few values naturally exceed the max of the first 300
    assert r["false_alarm_events"] <= 5


def test_no_alarms_during_probation():
    s = series(np.r_[np.full(100, 1.0), np.full(1900, 1.0)])
    s.iloc[10] = 1000
    al = alarms(zscore_detector(s))
    assert not al.iloc[:300].any()


def test_false_alarms_close_together_are_one_event():
    s = series(np.zeros(1000))
    al = pd.Series(False, index=s.index)
    al.iloc[[500, 501, 505]] = True               # within an hour -> 1 event
    al.iloc[900] = True                            # far away -> 2nd event
    assert evaluate(al, [])["false_alarm_events"] == 2


def test_mase_is_one_for_scale_error():
    assert mase(np.array([1.0, 1.0]), np.array([0.0, 0.0]), scale=1.0) == 1.0


def test_persistent_drops_single_point_blips():
    from capfc.detect import persistent
    al = series(np.zeros(10)).astype(bool)
    al.iloc[3] = True                              # 1-point blip
    al.iloc[6:8] = True                            # held for 2 points
    out = persistent(al, 2)
    assert list(out[out].index) == [al.index[7]]   # fires once the 2nd point confirms


def test_adaptive_threshold_follows_a_level_shift():
    from capfc.detect import adaptive_alarms
    rng = np.random.default_rng(1)
    # noise scale doubles a third of the way in, then stays: a frozen threshold keeps firing,
    # an adaptive one gets used to the new normal after a week of data
    n = 288 * 20
    v = rng.normal(0, 1, n)
    v[n // 3:] *= 2
    score = series(np.abs(v))
    frozen = alarms(score).iloc[-288 * 5:].sum()
    adaptive = adaptive_alarms(score).iloc[-288 * 5:].sum()
    assert adaptive < frozen / 3
