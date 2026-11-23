"""R1, raw features (naive): downsample every sensor curve to T' points, then flatten.

- GasWindTunnel  (N, 72, 1500) -> (N, 72, 100) -> F_raw (N, 7200)
- EverydayScents (N, 11, 210)  -> (N, 11, 70)  -> F_raw (N, 770)

Downsampling = mean of T' consecutive, (almost) equal-length blocks, so T' does not have to
divide T (needed for the C6 ablation: half and double T').
Nothing is fitted, so the features can be computed once for all samples.
"""
import numpy as np


def downsample(X, t_prime):
    """X: (N, C, T) -> (N, C, t_prime), mean of each block of time points."""
    T = X.shape[-1]
    edges = np.linspace(0, T, t_prime + 1).astype(int)               # (t_prime + 1,)
    sums = np.add.reduceat(X, edges[:-1], axis=-1)                   # (N, C, t_prime)
    return sums / np.diff(edges)                                     # (N, C, t_prime)


def raw_features(X, t_prime):
    """X: (N, C, T) -> F_raw (N, C * t_prime)."""
    Xd = downsample(X, t_prime)                                      # (N, C, t_prime)
    return Xd.reshape(len(X), -1)                                    # (N, C * t_prime)
