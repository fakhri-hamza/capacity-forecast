"""Anomaly detectors for a single metric, and an event-level evaluation.

All detectors are label-free. Each returns a score per point; the alarm threshold is set
from the first 15% of the series (NAB's "probationary period"), where the detector is
assumed to be learning what normal looks like. No labels are used to tune anything.

Evaluation (event level, like an on-call engineer would judge it):
  - a labelled anomaly window counts as DETECTED if at least one alarm falls inside it;
  - alarms outside windows are merged into events (alarms < 1 h apart = one event);
    each such event is one FALSE ALARM (one useless page in the middle of the night).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

DAY = 288                 # 5-minute points per day
PROBATION = 0.15


def zscore_detector(s: pd.Series) -> pd.Series:
    """Distance from the rolling mean of the past day, in rolling standard deviations."""
    past = s.shift(1).rolling(DAY, min_periods=DAY // 4)
    return ((s - past.mean()) / (past.std() + 1e-9)).abs()


def seasonal_detector(s: pd.Series) -> pd.Series:
    """Compare each point with the same time yesterday; robust z-score of that difference."""
    resid = s - s.shift(DAY)
    past = resid.shift(1).rolling(DAY, min_periods=DAY // 4)
    med = past.median()
    mad = (resid.shift(1) - med).abs().rolling(DAY, min_periods=DAY // 4).median()
    return ((resid - med) / (1.4826 * mad + 1e-9)).abs()


def iforest_detector(s: pd.Series) -> pd.Series:
    """Isolation Forest on simple local features, trained on the probationary period only."""
    f = pd.DataFrame({
        "value": s,
        "diff": s.diff(),
        "dev_1h": s - s.rolling(12, min_periods=1).mean(),
        "std_1h": s.rolling(12, min_periods=2).std(),
    }).fillna(0)
    n_train = int(len(s) * PROBATION)
    model = IsolationForest(n_estimators=200, random_state=0).fit(f.iloc[:n_train])
    return pd.Series(-model.score_samples(f), index=s.index)


DETECTORS = {
    "rolling z-score": zscore_detector,
    "seasonal (vs yesterday)": seasonal_detector,
    "isolation forest": iforest_detector,
}


def alarms(score: pd.Series, factor: float = 1.0) -> pd.Series:
    """Alarm when the score is higher than anything seen during the probationary period.

    Same rule for every detector, no tuning on labels: "more unusual than anything in the
    first 15% of the data". `factor` scales that threshold to trade detections for
    fewer false alarms (used only to draw the trade-off curve).
    """
    n_train = int(len(score) * PROBATION)
    ref = score.iloc[:n_train].replace([np.inf, -np.inf], np.nan).dropna()
    threshold = factor * ref.max() if len(ref) else np.inf
    out = (score > threshold).fillna(False)
    out.iloc[:n_train] = False
    return out


def evaluate(alarm: pd.Series, windows: list) -> dict:
    t = alarm.index
    start = t[int(len(t) * PROBATION)]
    windows = [(a, b) for a, b in windows if b > start]
    in_window = np.zeros(len(t), dtype=bool)
    detected = 0
    for a, b in windows:
        mask = (t >= a) & (t <= b)
        in_window |= mask
        detected += bool(alarm[mask].any())
    false_times = t[alarm.to_numpy() & ~in_window]
    false_events = 0
    if len(false_times):
        gaps = np.diff(false_times.values).astype("timedelta64[m]").astype(int)
        false_events = 1 + int((gaps > 60).sum())
    days = (t[-1] - start).total_seconds() / 86400
    return {"windows": len(windows), "detected": detected,
            "false_alarm_events": false_events, "days": days}


# ---------------------------------------------------------------------- alert policies
# The detectors above produce a score; a *policy* turns scores into pages. Two label-free ideas
# to cut false alarms without touching the detector:
#   adaptive threshold  "more unusual than 99.9% of the last 7 days" instead of a threshold frozen
#                       after the first 15% (metrics drift: what was extreme in week 1 is normal later)
#   persistence         only page when the condition holds for k consecutive points (10 min for k=2),
#                       which filters one-sample blips


def adaptive_alarms(score: pd.Series, factor: float = 1.0, q: float = 0.999,
                    window: int = 7 * DAY) -> pd.Series:
    """Alarm when the score exceeds factor x the q-quantile of the previous `window` points.

    Until a full window is available the quantile uses all past points (at least the
    probationary period). Same no-labels rule as `alarms`; `factor` only draws the trade-off.
    """
    n_train = int(len(score) * PROBATION)
    past = score.replace([np.inf, -np.inf], np.nan).shift(1)
    threshold = factor * past.rolling(window, min_periods=max(n_train, 1)).quantile(q)
    out = (score > threshold).fillna(False)
    out.iloc[:n_train] = False
    return out


def persistent(alarm: pd.Series, k: int = 2) -> pd.Series:
    """Keep an alarm only once it has held for k consecutive points."""
    if k <= 1:
        return alarm
    return alarm.astype(int).rolling(k, min_periods=k).sum().eq(k)
