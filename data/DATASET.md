# Datasets: the bottom line

Both datasets end up in the same form, one 3-D array per dataset:

```
X_signals.npy: (recordings, sensor channels, time points)
GasWindTunnel  (600, 72, 1500)
EverydayScents (144, 11, 210)
```

Each curve has two parts:
- **baseline:** clean air, before the gas or scent;
- **exposure:** the gas or scent is present.

The phase after it (recovery in GasWindTunnel, cleaning in EverydayScents) is **dropped in both datasets**, so a classifier only sees what a detector sees while the gas is present.

---

## 1. GasWindTunnel (UCI 251), as used in this project

### 1.1 The structure

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

### 1.2 The 10 gases (one class each)

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

### 1.3 The 72 channels

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
- **A value** is the raw electrical reading of the sensor (no unit given), after baseline correction (section 1.5).

### 1.4 Time axis: from the raw file to 1500 points

A raw file covers 260 s with about 13,000 to 26,000 rows at an uneven rate. It has three phases:

| Phase | Raw time | Kept? |
|---|---|---|
| Baseline | 0 → release start (about 20.2 s) | yes → **150 points** |
| Exposure | release start → release end (about 20.2 → 200.2 s, 180 s) | yes → **1350 points** |
| Recovery | release end → 260 s (about 60 s, gas off) | **no** |

The release start and end are read from the gas flow column of **each recording**. They vary by only ±0.06 s, so every cut is made at that recording's own times.

**Why the length is still 1500 after dropping the recovery phase.** The 1500 is not a count of raw rows. It is a number we choose, and each kept phase is **resampled** (linear interpolation) onto that many evenly spaced points:

```
before (first version):  0 s ─────────────── 260 s      →  1500 points, 0.173 s apart
                          baseline 116 | exposure 1039 | recovery 345

now:                     0 s ── 20.2 s                  →   150 points, 0.135 s apart
                         20.2 s ────────── 200.2 s      →  1350 points, 0.133 s apart
                         200.2 s → 260 s                →  dropped
                                                           ─────────
                                                           1500 points
```

The 345 recovery points did not disappear from the end of the array. The same 1500 points are spread over 200 s instead of 260 s, so the points are slightly denser (0.133 s instead of 0.173 s apart).

- 150 / 1350 follows the real time split (20.2 s of the 200.2 s kept is about 10%), so both phases have nearly the same spacing.
- It keeps the shape `(600, 72, 1500)` from the plan.
- The gas starts at **point 150 in every recording**, and the curve ends when the gas is switched off.

| Points | Nominal time | Phase |
|---|---|---|
| 0 – 149 | 0 – 20.2 s | baseline (clean air) |
| 150 – 1499 | 20.2 – 200.2 s | exposure (gas flowing) |

The nominal time of every point is stored in `info.json` (`time_s`).

### 1.5 Baseline correction, step by step

For every recording `i` and every sensor `c`:

```
1. resting level   L[i, c] = mean of the 150 baseline points  X_raw[i, c, 0:150]
2. corrected curve X[i, c, :] = X_raw[i, c, :] - L[i, c]          (all 1500 points)
3. keep L separately: baseline_level.npy, shape (600, 72)
```

- After the correction, every curve averages 0 over its baseline. The exposure part shows how much the sensor moved because of the gas.
- **Why:** sensors drift, and identical sensors on different boards rest at different levels (for example 392 vs 258). Subtracting the resting level removes that offset, so the classifier sees the response, not the sensor's starting point.
- One constant is subtracted per sensor, so the shape of the curve does not change.
- **Why L is kept:** R2 uses it as feature 1 ("baseline level") and for the normalized response (feature 3). The baseline only check (C5) adds it back, because a resting level that differs between gases would be a leak.
- **Known glitch:** the very first raw point (t = 0 s) is often a startup spike, in about 29% of recordings. It is 1 of the 150 baseline points.

### 1.6 Fixed settings (the same for all 600 recordings)

- **Board location:** L3 (middle distance from the source, out of L1–L6).
- **Heater voltage:** 500 V setting (middle of 400–600). It sets the sensor temperature, which changes how strongly each sensor reacts to each gas.
- **What varies:** gas, wind speed (fan 0 / 60 / 100) and repetition.

### 1.7 Labels and folds

- **`y_multi`:** shape (600,), 10 classes, 60 per class.
- **`y_binary`:** shape (600,), CO vs other gases, 60 positive / 540 negative.
- **Folds:** `StratifiedKFold(5)`, stratified by gas × wind speed (30 groups of 20). That gives 480 training and 120 test recordings per fold, with 12 test recordings per gas.

### 1.8 Files (`data/processed/gas_wind_tunnel/`)

| File | Content |
|---|---|
| `X_signals.npy` | (600, 72, 1500) float32, baseline corrected, not scaled |
| `baseline_level.npy` | (600, 72) resting level per sensor |
| `meta.csv` | 600 rows: gas, y_multi, y_binary, fan, wind_id, strat, rep, release start / end, source file |
| `info.json` | T = 1500, n_base = 150, exposure_end = 1500, time_s, location, voltage, folders, class names |

---

## 2. EverydayScents (our own data)

### 2.1 The structure

```
144 samples (from 81 sessions)
 = ambient 33 + coffee 39 + truffle 50 + black_pepper 22

1 sample = 11 gas channels × 210 time points

X_signals.npy: (144, 11, 210)
```

| | Rows (sensor curves) | Columns |
|---|---|---|
| All samples | 144 × 11 = **1,584** | 210 time points |
| One sample | **11** | 210 time points |
| One channel, all samples | **144** | 210 time points |

**Where 144 comes from:**
- The session files contain 172 labelled samples.
- Label cleaning (00b) merges the raw labels into 4 classes and drops 17 test / unrelated samples, leaving **155**.
- Segmentation (00c) drops **11 aborted or restarted recordings**, whose sampling phase lasts only 0–15 s (or the file ends before it starts). That leaves **144**.

### 2.2 The classes

| Class | Raw labels merged | Samples |
|---|---|---|
| ambient | ambient | 33 |
| coffee | coffee, Arabica_coffee, Robusta_coffee, coffee_outdoor, coffee ground, coffee_ground_heated | 39 |
| truffle | truffle, fresh_truffle, fresh_truffle_with_jar, truffle_outdoor | 50 |
| black_pepper | black_pepper, black_paper (typo) | 22 |

### 2.3 The 11 channels

Unlike GasWindTunnel, the 11 channels are **11 different sensor types**, one per channel, named after the gas each one targets: `hcho, ch4, voc, odor, etoh, h2s, no2, nh3, co, smoke, h2`. As with MOX sensors in general, each also reacts to other compounds, so a scent is again recognised from the pattern across channels.

- **Value used:** `voltage_v`, the sensor output in volts. It is used instead of `raw` because it does not depend on the amplifier gain setting, which changed between sessions.
- **Not used as features:** `gas_resistance_ohm`, `temperature_c`, `humidity_percent`, `pressure_hpa`.

### 2.4 Time axis: from the session file to 210 points

**How a sample is cut out of its session.**
- A session file holds one or more samples, in long format (one row per reading: timestamp, phase, sensor, value, …).
- Each sample runs through baseline → sampling → cleaning.
- The sample's own phase times come from its `phase_history` in the metadata.
- Its rows are cut out by phase and time window, then pivoted to wide format (11 channels × raw time).

| Phase | Typical raw length | Raw readings (about one every 1.5 s) | Kept? |
|---|---|---|---|
| Baseline (`BASELINE_MEASUREMENT`) | about 180 s (57 – 299 s) | about 120 | yes → **60 points** |
| Sampling (`SAMPLE_MEASUREMENT`) | about 300 s (31 – 599 s) | about 200 | yes → **150 points** |
| Cleaning (`CLEANING`) | about 300 s | | **no** |

Each phase is resampled on its own to a fixed number of points, exactly like GasWindTunnel. In both datasets the length is chosen, not counted:

```
baseline  (about 180 s, ~120 readings) → 60 points   (about 3.0 s apart)
sampling  (about 300 s, ~200 readings) → 150 points  (about 2.0 s apart)
cleaning                               → dropped
                                         ─────────
                                         210 points
```

| Points | Phase |
|---|---|
| 0 – 59 | baseline (clean air) |
| 60 – 209 | sampling (scent present) |

- **Phase lengths are not all the same.** Most samples have 180 s + 300 s, but some are longer or shorter: sampling ranges from 31 to 599 s. Every phase is still stretched or squeezed to the same number of points.
- So point 100 is "40% into the sampling phase", not a fixed number of seconds. GasWindTunnel does not have this issue, because all its phases have the same length.

### 2.5 Baseline correction, step by step

The logic is the same as for GasWindTunnel. For every sample `i` and every channel `c`:

```
1. resting level   L[i, c] = mean of the 60 baseline points  X_raw[i, c, 0:60]
2. corrected curve X[i, c, :] = X_raw[i, c, :] - L[i, c]        (all 210 points)
3. keep L separately: baseline_level.npy, shape (144, 11)
```

- The resting level matters even more here. Samples were recorded on different days, indoors and outdoors, with different humidity and temperature, and the sensors drift between sessions.
- Without the correction, a model could learn "which day was this" instead of "which scent was this". Black pepper, for example, was recorded on only 4 days.
- The baseline only check (C5) tests exactly this.

### 2.6 Labels and folds

- **`y_multi`:** shape (144,), 4 classes.
- **`y_binary`:** shape (144,), truffle vs not truffle, 50 / 94.
- **Folds:** `StratifiedGroupKFold(5)` with groups = session, so all samples of a session stay on the same side. That gives about 115 training and 29 test samples per fold, with at least 3 black_pepper samples in every test fold.

### 2.7 Files (`data/processed/everyday_scents/`)

| File | Content |
|---|---|
| `X_signals.npy` | (144, 11, 210) float32, baseline corrected, not scaled |
| `baseline_level.npy` | (144, 11) resting level per channel |
| `meta.csv` | 144 rows: session_id, sample_id, day, raw_label, class, y_multi, y_binary, phase times, raw point counts |
| `info.json` | T = 210, n_base = 60, exposure_end = 210, channels, class names |

---

## 3. Shape of each representation (input to the classification heads)

Every representation turns one recording (C, T) into one vector, giving a matrix of shape (N, D).

| Representation | Per sensor curve | GasWindTunnel F | EverydayScents F |
|---|---|---|---|
| R1 raw (naive) | average blocks of points → 100 (GWT, blocks of 15) / 70 (ES, blocks of 3) points | (600, 72 × 100) = **(600, 7200)** | (144, 11 × 70) = **(144, 770)** |
| R2 handcrafted | 12 numbers: baseline level, steady state, normalized response, peak, time to peak, rise slope, decay slope, area, 4 EMA features | (600, 72 × 12) = **(600, 864)** | (144, 11 × 12) = **(144, 132)** |
| R3 encoder | 1D CNN → 32 learned numbers | (600, 72 × 32) = **(600, 2304)** | (144, 11 × 32) = **(144, 352)** |

- R1 and R2 are fixed formulas, computed once.
- R3 is learned: the encoder is retrained inside every fold, on the training recordings only.
- **The two datasets are never merged:** they have different sensors (72 vs 11 channels).
