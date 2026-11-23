# Datasets: the bottom line

## GasWindTunnel (UCI 251), as used in this project

### The structure

```
600 recordings
 = 10 gases × 60 recordings per gas
 = 10 gases × 3 wind speeds × 20 repetitions

1 recording = 72 sensor channels × 1500 time points
 72 channels = 9 boards × 8 metal-oxide (MOX) sensors

X_signals.npy: (600, 72, 1500)
```

Flattened to one row per sensor curve (the table view at the end of notebook 00a):

| | Rows (sensor curves) | Columns |
|---|---|---|
| All recordings | 600 × 72 = **43,200** | 1500 time points |
| One gas | 60 × 72 = **4,320** | 1500 time points |
| One recording | **72** | 1500 time points |
| One sensor, all recordings | **600** | 1500 time points |

### The 10 gases (one class each)

| Gas | Concentration used |
|---|---|
| Acetaldehyde | 500 ppm |
| Acetone | 2500 ppm |
| Ammonia | 10000 ppm |
| Benzene | 200 ppm |
| Butanol | 100 ppm |
| CO | 4000 ppm (the 1000 ppm folder is incomplete) |
| Ethylene | 500 ppm |
| Methane | 1000 ppm |
| Methanol | 200 ppm |
| Toluene | 200 ppm |

Each gas has one fixed concentration, so the task is "which gas", not "how much".

### The 72 channels

All 9 boards carry the same 8 Figaro MOX sensors. The column names `B<board>_S<position>` (for example `B1_S1`) were made by us; the raw files have no headers.

| Position | Sensor | Designed for |
|---|---|---|
| S1 | TGS2611 | methane |
| S2 | TGS2612 | methane, LP gases |
| S3 | TGS2610 | LP gas (propane, butane) |
| S4 | TGS2600 | air contaminants (hydrogen, CO) |
| S5, S6 | TGS2602 | air quality: VOCs, ammonia, H₂S, toluene |
| S7, S8 | TGS2620 | solvent vapours, alcohols |

> The position → sensor order comes from the UCI description and has not yet been checked against the source.

- **No sensor is specific to one gas.** Every sensor reacts to most of the 10 gases with a different strength. The class is recognised from the pattern across all 72 channels (the fingerprint).
- **The 9 boards are 9 copies at different positions** across the wind tunnel at one distance from the gas source. They see different parts of the gas plume.
- **A value** is the raw electrical reading of the sensor (no unit given), after baseline correction (see below).

### One recording over time (1500 points over 260 s, one point ≈ 0.173 s)

| Phase | Time | Points | What happens |
|---|---|---|---|
| Baseline | 0 – 20.1 s | 0 – 115 (116 points) | clean air, no gas |
| Exposure | 20.2 – 200.2 s | 116 – 1154 | gas released |
| Recovery | 200.2 – 260 s | 1155 – 1499 | gas off, sensors return |

- The raw files have about 13,000 to 26,000 rows at an uneven rate. They are resampled to 1500 evenly spaced points, so every recording has the same length.
- **Baseline correction:** the mean of the baseline (first 116 points) is subtracted from every sensor. A curve therefore starts near 0 and shows the change caused by the gas.
- The resting level that was subtracted is kept in `baseline_level.npy`, shape (600, 72). The handcrafted features (R2) and the baseline only check (C5) use it.
- **Known glitch:** the first point (t = 0 s) is often a startup spike, in about 29% of recordings.

### Fixed settings (the same for all 600 recordings)

- **Board location:** L3 (middle distance from the source, out of L1–L6).
- **Heater voltage:** 500 V setting (middle of 400–600). It sets the sensor temperature, which changes how strongly each sensor reacts to each gas.
- **What varies:** gas, wind speed (fan 0 / 60 / 100) and repetition.

### Labels and folds

- **`y_multi`:** shape (600,), 10 classes, 60 per class.
- **`y_binary`:** shape (600,), CO vs other gases, 60 positive / 540 negative.
- **Folds:** `StratifiedKFold(5)`, stratified by gas × wind speed (30 groups of 20). That gives 480 training and 120 test recordings per fold, with 12 test recordings per gas.

### Shape of each representation (input to the classification heads)

Every representation turns one recording (72, 1500) into one vector, giving a matrix of shape (600, D).

| Representation | Per sensor curve (1500 points) | Per recording | Feature matrix F |
|---|---|---|---|
| R1 raw (naive) | average every 15 points → 100 points | (72, 100) | **(600, 7200)** |
| R2 handcrafted | 12 numbers (baseline level, steady state, peak, slopes, area, …) | (72, 12) | **(600, 864)** |
| R3 encoder | 1D CNN → 32 learned numbers | (72, 32) | **(600, 2304)** |

R1 and R2 are fixed formulas, computed once. R3 is learned: the encoder is retrained inside every fold, on the 480 training recordings only.

### Files (`data/processed/gas_wind_tunnel/`)

| File | Content |
|---|---|
| `X_signals.npy` | (600, 72, 1500) float32, baseline corrected, not scaled |
| `baseline_level.npy` | (600, 72) resting level per sensor |
| `meta.csv` | 600 rows: gas, y_multi, y_binary, fan, wind_id, strat, rep, release times, source file |
| `info.json` | T, n_base = 116, exposure_end = 1155, time step, location, voltage, folders, class names |

---

## EverydayScents (our own data), for comparison

| | |
|---|---|
| Recordings | **144** samples from 81 sessions (155 labelled; 11 aborted recordings dropped) |
| Classes | ambient 33, coffee 39, truffle 50, black_pepper 22 |
| Channels | 11 gas channels (hcho, ch4, voc, odor, etoh, h2s, no2, nh3, co, smoke, h2), each a different sensor type |
| Time | baseline resampled to 60 points + sampling phase resampled to 150 points = 210 (no recovery phase) |
| `X_signals.npy` | **(144, 11, 210)** |
| Binary task | truffle vs not truffle (50 / 94) |
| Folds | `StratifiedGroupKFold(5)` grouped by session: about 115 train / 29 test |
| R1 / R2 / R3 | (144, 770) / (144, 132) / (144, 352) |

The two datasets are never merged: they have different sensors (72 vs 11 channels).
