"""EverydayScents loader: label map and segmentation of the session files.

Will provide:
- reading session_<id>_metadata.json (samples, labels, phase durations, gains) for the 81 sessions;
- the raw label -> class map (ambient, coffee, truffle, black_pepper) and the dropped labels
  (test*, a, burned_soil, forest_soil), giving 155 usable samples;
- cutting each sample out of session_<id>.csv (long format: timestamp, phase, sensor, value, raw,
  voltage_v, gain) by phase and time window and pivoting it to wide, shape (11, T_raw),
  using the voltage_v column for the 11 gas channels
  (hcho, ch4, voc, odor, etoh, h2s, no2, nh3, co, smoke, h2);
- labels y_multi (155,), y_binary (155,) for truffle vs not truffle, and session_id (155,) for grouping.

gas_resistance_ohm, temperature_c, humidity_percent and pressure_hpa are not used as features.
"""
