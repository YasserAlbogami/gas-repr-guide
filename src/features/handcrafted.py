"""R2, hand feature engineering: k = 12 features per sensor, one function per feature.

Every function takes one baseline-corrected curve r (T,) (response relative to the baseline)
and the time indices of the phases:
    0 .. n_base-1          baseline (before the gas / scent)
    n_base .. exp_end-1    exposure
    exp_end .. T-1         recovery (GasWindTunnel only; EverydayScents ends with the exposure)

 #  feature                 description
 1  baseline level          mean of the baseline (the resting level, taken before baseline correction)
 2  steady state response   mean of the last 20% of the exposure (minus baseline: r is already corrected)
 3  normalized response     steady state response / baseline level
 4  peak response           largest change from baseline during the exposure (signed)
 5  time to peak            time points from the start of the exposure to the peak
 6  rise slope              slope of a line fitted from the start of the exposure to the peak
 7  decay slope             slope of a line fitted from the peak to the end of the curve
 8  area under the curve    sum of the response during the exposure (one point = one time step)
 9  EMA max, alpha 0.1      max of the exponential moving average of the derivative
10  EMA min, alpha 0.1      min of the same
11  EMA max, alpha 0.01     same with stronger smoothing
12  EMA min, alpha 0.01

Shapes: X (N, C, T) -> (N, C, 12) -> F_hand (N, C * 12)
    GasWindTunnel (600, 864), EverydayScents (144, 132)
subset=STEADY_STATE keeps features 1 to 4 only (C6 ablation): (N, C * 4).
Nothing is fitted, so the features can be computed once for all samples.
"""
import numpy as np
from scipy.signal import lfilter

FEATURE_NAMES = ["baseline_level", "steady_state", "normalized", "peak", "time_to_peak",
                 "rise_slope", "decay_slope", "auc",
                 "ema_max_0.1", "ema_min_0.1", "ema_max_0.01", "ema_min_0.01"]
STEADY_STATE = FEATURE_NAMES[:4]


def baseline_level(level):
    """1. The resting level of the sensor (a number computed before baseline correction)."""
    return level


def steady_state(r, n_base, exp_end):
    """2. Mean of the last 20% of the exposure."""
    n = max(1, (exp_end - n_base) // 5)
    return r[exp_end - n:exp_end].mean()


def normalized(r, n_base, exp_end, level):
    """3. Steady state response relative to the resting level."""
    return steady_state(r, n_base, exp_end) / (abs(level) + 1e-6)


def peak_index(r, n_base, exp_end):
    """Index (in the whole curve) of the largest absolute change during the exposure."""
    return n_base + int(np.argmax(np.abs(r[n_base:exp_end])))


def peak(r, n_base, exp_end):
    """4. Signed value of the largest change from baseline during the exposure."""
    return r[peak_index(r, n_base, exp_end)]


def time_to_peak(r, n_base, exp_end):
    """5. Time points from the start of the exposure to the peak."""
    return peak_index(r, n_base, exp_end) - n_base


def _slope(segment):
    """Slope of a least-squares line through a segment (0 if it has fewer than 2 points)."""
    if len(segment) < 2:
        return 0.0
    return np.polyfit(np.arange(len(segment)), segment, 1)[0]


def rise_slope(r, n_base, exp_end):
    """6. Slope from the start of the exposure to the peak."""
    return _slope(r[n_base:peak_index(r, n_base, exp_end) + 1])


def decay_slope(r, n_base, exp_end):
    """7. Slope from the peak to the end of the curve."""
    return _slope(r[peak_index(r, n_base, exp_end):])


def auc(r, n_base, exp_end):
    """8. Area under the response during the exposure."""
    return r[n_base:exp_end].sum()


def ema_derivative(r, n_base, alpha):
    """Exponential moving average of the derivative from the start of the exposure:
    y[k] = (1 - alpha) * y[k-1] + alpha * (r[k] - r[k-1])"""
    d = np.diff(r[n_base - 1:]) if n_base > 0 else np.diff(r)       # (T - n_base,)
    return lfilter([alpha], [1, -(1 - alpha)], d)


def ema_features(r, n_base):
    """9-12. Max and min of the smoothed derivative for alpha 0.1 and 0.01."""
    out = []
    for alpha in (0.1, 0.01):
        y = ema_derivative(r, n_base, alpha)
        out += [y.max(), y.min()]
    return out


def curve_features(r, level, n_base, exp_end):
    """All 12 features of one curve r (T,); returns (12,)."""
    return np.array([
        baseline_level(level),
        steady_state(r, n_base, exp_end),
        normalized(r, n_base, exp_end, level),
        peak(r, n_base, exp_end),
        time_to_peak(r, n_base, exp_end),
        rise_slope(r, n_base, exp_end),
        decay_slope(r, n_base, exp_end),
        auc(r, n_base, exp_end),
        *ema_features(r, n_base),
    ])


def handcrafted_features(X, levels, n_base, exp_end, subset=None):
    """X: (N, C, T) baseline-corrected curves, levels: (N, C) resting levels.

    returns F_hand (N, C * k), k = 12 (or len(subset))
    """
    N, C, _ = X.shape
    F = np.empty((N, C, len(FEATURE_NAMES)))                         # (N, C, 12)
    for i in range(N):
        for c in range(C):
            F[i, c] = curve_features(X[i, c].astype(float), float(levels[i, c]), n_base, exp_end)
    if subset is not None:
        F = F[:, :, [FEATURE_NAMES.index(f) for f in subset]]        # (N, C, k)
    return F.reshape(N, -1)                                          # (N, C * k)
