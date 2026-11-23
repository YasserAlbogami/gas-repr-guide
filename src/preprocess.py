"""Preprocessing shared by all representations (identical within a dataset).

- resample: put a curve on a fixed number of evenly spaced points
  GasWindTunnel (72, T_raw) -> (72, 1500)
  EverydayScents baseline phase -> 60 points, sampling phase -> 150 points, joined to (11, 210)
- baseline_correct: subtract the per-sensor mean of the baseline part

Scaling is NOT done here: the StandardScaler is fitted on the training fold only, in cv.py.
"""
import numpy as np


def resample(x, t, n_points, t_start=None, t_end=None):
    """Linear interpolation of every sensor onto n_points evenly spaced times.

    x: (C, T_raw) sensor values, t: (T_raw,) time stamps (increasing)
    t_start, t_end: time range of the new grid (default: first and last time stamp)
    returns (C, n_points)
    """
    t_start = t[0] if t_start is None else t_start
    t_end = t[-1] if t_end is None else t_end
    grid = np.linspace(t_start, t_end, n_points)                      # (n_points,)
    out = np.empty((x.shape[0], n_points))                            # (C, n_points)
    for c in range(x.shape[0]):
        out[c] = np.interp(grid, t, x[c])
    return out


def baseline_correct(x, n_base):
    """Subtract, for every sensor, the mean of its first n_base points.

    x: (C, T) or (N, C, T); returns the same shape
    """
    base = x[..., :n_base].mean(axis=-1, keepdims=True)              # (..., C, 1)
    return x - base
