"""R1, raw features (naive).

Will provide downsampling of each sensor curve to T' points followed by flattening:
- GasWindTunnel  (N, 72, 1500) -> (N, 72, 100) -> F_raw (N, 7200)
- EverydayScents (N, 11, 210)  -> (N, 11, 70)  -> F_raw (N, 770)
T' is a parameter so the C6 ablation can use half and double the default.
Nothing is fitted, so the same function can be applied to train and test data.
"""
