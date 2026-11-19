"""Preprocessing shared by all representations (identical within a dataset).

Will provide:
- resampling a curve to a fixed length:
  GasWindTunnel (72, T_raw) -> (72, 1500);
  EverydayScents baseline phase -> 60 points and sampling phase -> 150 points, joined to (11, 210);
- baseline correction per sensor: subtract the mean before the gas release (GasWindTunnel)
  or the mean of the baseline phase (EverydayScents).

Scaling is NOT done here: the StandardScaler is fitted on the training fold only, in cv.py.
"""
