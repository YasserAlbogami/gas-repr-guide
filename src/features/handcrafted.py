"""R2, hand feature engineering: k = 12 features per sensor, one function per feature.

Features, computed on the response relative to the baseline:
 1 baseline level           mean of the baseline
 2 steady state response    mean of the last part of the exposure minus baseline
 3 normalized response      steady state response / baseline
 4 peak response            maximum change from baseline
 5 time to peak             time step of the peak
 6 rise slope               slope of a line fitted on the rising part
 7 decay slope              slope of a line fitted after the peak
 8 area under the curve     integral of the response during exposure
 9-12 EMA features          max and min of a smoothed derivative at two smoothing values

Shapes: (N, C, T) -> (N, C, 12) -> F_hand (N, C * 12)
- GasWindTunnel  (600, 864)
- EverydayScents (155, 132)
A subset option (features 1 to 4) supports the C6 ablation.
"""
