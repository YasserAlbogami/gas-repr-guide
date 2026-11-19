"""GasWindTunnel loader (UCI 251, Gas Sensor Arrays in Open Sampling Settings).

Will provide:
- reading one recording file into an array of shape (72, T_raw): 9 boards x 8 metal oxide sensors;
- selecting the subset used in the project: one board location and one heater voltage
  (chosen in 00a), all 3 wind speeds, 10 gases x 20 repetitions = 600 recordings;
- building the label vectors y_multi (600,) with 10 gas classes and y_binary (600,) for CO vs other,
  and the stratification id gas x wind speed.

Only the chosen location and voltage are read, because the full download is very large.
"""
