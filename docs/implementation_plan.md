# Raw, Engineered, or Learned? A Feature Representation Guide for Gas Sensor Channels

## Implementation Plan (v3: two datasets, three representations)

---

## 1. The problem, concretely

### What an electronic nose does

An electronic nose works like a scent detection model. When a scent reaches the device, for example black coffee, each sensor reacts to it. Each sensor gives **one voltage signal over time**. Sensors react differently: some are tuned to a gas family and react strongly, while broad sensors react a little to many gases. Together, the sensor curves form a **fingerprint**, and a model learns to map that fingerprint to a label such as "coffee". The model never measures each compound separately; it only sees the overall response pattern.

### One recording

- One gas or one scent is presented to the sensors. **Only that class is present**; all other classes are absent.
- Every sensor records one signal over T time steps.
- The recording is labeled with the presented class.

```
recording #17
input : C sensor curves × T time steps
label : Acetone    (only acetone was released; the other classes are absent)
```

### What we predict

**Given the sensor curves of a recording, predict which gas or scent was presented.** Each dataset has one multiclass task and one binary task (Section 3).

### What the project studies

Before a classifier can be trained, the C × T block has to be turned into a feature vector. We compare three ways:

1. **Raw features (naive):** use the sensor curves directly.
2. **Hand feature engineering:** compute standard response features per sensor.
3. **Encoder feature maps:** a small 1D CNN learns the features.

We fit the **same classification heads** on each representation, with the same folds and preprocessing, on **two contrasting datasets**. The outcome is a **feature representation guide for gas sensor channels for downstream tasks**: practical advice for people building gas detection or scent detection classifiers.

### Contributions

1. A controlled comparison of raw, handcrafted and encoder representations under identical heads, folds and preprocessing.
2. Two contrasting settings: a large lab dataset of pure gases and a small real world dataset of everyday scents.
3. A practical guide: which representation to choose depending on data size, task and resources, backed by the results.
4. A new real world scent dataset, EverydayScents.

### Scope of claims

- Results hold for these two datasets and these sensor types; we do not claim one representation is best in general.
- None of the three representations is new; the contribution is the controlled comparison and the guide.
- EverydayScents is small, so its results are reported with their uncertainty.

---

## 2. Datasets

### Dataset A: GasWindTunnel (pure gases, lab, large)

Public dataset: *Gas Sensor Arrays in Open Sampling Settings*, UCI 251.

| Property | Value |
|---|---|
| Recordings (full) | 18,000 |
| Sensors | 72 metal oxide sensors (9 boards × 8 sensors) |
| Sampling rate | 100 Hz |
| Classes | 10 gases: Acetaldehyde, Acetone, Ammonia, Benzene, Butanol, CO, Ethylene, Methane, Methanol, Toluene |
| Conditions | 6 board locations × 3 wind speeds × 5 heater voltages, 20 repetitions each |

**Subset used:** one board location and one heater voltage (chosen in the audit), all 3 wind speeds:
10 gases × 3 wind speeds × 20 repetitions = **600 recordings, 60 per gas**.

### Dataset B: EverydayScents (real scents, field and lab, small)

Our own dataset, collected during the ICTP internship with a low cost electronic nose.

| Property | Value |
|---|---|
| Sessions | 81 (2026-07-07 to 2026-08-07) |
| Usable samples after label cleaning | 155 |
| Gas channels used as features | 11: hcho, ch4, voc, odor, etoh, h2s, no2, nh3, co, smoke, h2 |
| Not used as features | gas_resistance_ohm, temperature_c, humidity_percent, pressure_hpa |
| Sampling | about one reading every 1.75 s per sensor |
| Protocol per sample | baseline → sampling → cleaning (most samples: 180 s → 300 s → 300 s) |
| Files | `session_<id>.csv` (long format: timestamp, phase, sensor, value, raw, voltage_v, gain) + `session_<id>_metadata.json` (samples, labels, phase durations, gains) |

**Label cleaning (raw label → class):**

| Class | Raw labels merged | Samples | Sessions | Days |
|---|---|---|---|---|
| ambient | ambient | 37 | 14 | 9 |
| coffee | coffee, Arabica_coffee, Robusta_coffee, coffee_outdoor, coffee ground, coffee_ground_heated | 42 | 20 | 11 |
| truffle | truffle, fresh_truffle, fresh_truffle_with_jar, truffle_outdoor | 54 | 20 | 10 |
| black_pepper | black_pepper, black_paper (typo) | 22 | 8 | 4 |
| *dropped* | test*, a, burned_soil, forest_soil | 17 | | |

**Open item:** confirm whether `old_dataset.csv` and `new_truffle_dataset.csv` overlap with the session files. The plan uses the session files only.

The two datasets have different sensors (72 vs 11 channels), so they are **never merged**. The same pipeline runs on each one separately.

---

## 3. Inputs, labels and tasks

| | GasWindTunnel | EverydayScents |
|---|---|---|
| X_signals | (600, 72, 1500) | (155, 11, 210) |
| Multiclass task | Which gas? 10 classes | Which scent? 4 classes: ambient, coffee, truffle, black_pepper |
| y_multi | (600,), 60 per class | (155,), 37 / 42 / 54 / 22 |
| Binary task | CO vs other gases | truffle vs not truffle |
| y_binary | (600,), 60 positive / 540 negative | (155,), 54 positive / 101 negative |
| Fold grouping | stratified by gas × wind speed | grouped by session, stratified by class |

**Time axis of EverydayScents:** phases have different durations across samples, so each sample is built from its baseline phase resampled to 60 points and its sampling phase resampled to 150 points: T = 60 + 150 = 210. The cleaning phase is not used.

---

## 4. Preprocessing (identical for all representations within a dataset)

| Step | GasWindTunnel | EverydayScents |
|---|---|---|
| 1. Load | one recording file → (72, T_raw) | segment each sample from its session CSV by phase and time window, pivot long → wide (11, T_raw) |
| 2. Signal value | sensor reading | `voltage_v` (independent of the gain setting) |
| 3. Resample | to T = 1500 | baseline → 60, sampling → 150, T = 210 |
| 4. Baseline correction | subtract the mean before the gas release, per sensor | subtract the mean of the baseline phase, per sensor |
| 5. Scaling | fitted on the training fold only | fitted on the training fold only |

### Leakage rules

1. EverydayScents is split **by session**; samples from one session never appear in both train and test.
2. Every fitted object (scaler, encoder, head) is fitted on training data only, inside each fold.
3. Hyperparameters are chosen on an inner validation split of the training data.

---

## 5. The three representations

Each representation turns one recording (C, T) into one vector. F has shape (N, D).

### R1. Raw features (naive)

Downsample each sensor curve to T' points, then flatten.

| | GasWindTunnel | EverydayScents |
|---|---|---|
| Input | (72, 1500) | (11, 210) |
| Downsampled | (72, 100) | (11, 70) |
| **F_raw** | **(600, 7200)** | **(155, 770)** |

### R2. Hand feature engineering

k = 12 features per sensor, computed on the response relative to the baseline:

| # | Feature | Description |
|---|---|---|
| 1 | Baseline level | mean of the baseline |
| 2 | Steady state response | mean of the last part of the exposure minus baseline |
| 3 | Normalized response | steady state response divided by baseline |
| 4 | Peak response | maximum change from baseline |
| 5 | Time to peak | time step of the peak |
| 6 | Rise slope | slope of a line fitted on the rising part |
| 7 | Decay slope | slope of a line fitted after the peak |
| 8 | Area under the curve | integral of the response during exposure |
| 9 to 12 | Exponential moving average features | max and min of a smoothed derivative at two smoothing values |

| | GasWindTunnel | EverydayScents |
|---|---|---|
| Per sensor | (72, 12) | (11, 12) |
| **F_hand** | **(600, 864)** | **(155, 132)** |

### R3. Encoder feature maps

A small 1D CNN with shared weights is applied to each sensor curve separately:

| Layer | GasWindTunnel (one sensor) | EverydayScents (one sensor) |
|---|---|---|
| Input | (1, 1500) | (1, 210) |
| Conv1d(1 → 16, k 7, stride 2, pad 3) + ReLU | (16, 750) | (16, 105) |
| Conv1d(16 → 32, k 5, stride 2, pad 2) + ReLU | (32, 375) | (32, 53) |
| Global average pooling | (32,) | (32,) |
| **F_enc** (all sensors flattened) | **(600, 2304)** | **(155, 352)** |

**Training:** inside each fold, on training data only, the encoder plus a temporary linear layer is trained on the class labels. The linear layer is removed, the encoder is frozen and produces the features, and the heads are fitted on them exactly as for R1 and R2.

**Reference point:** the same CNN trained end to end as a classifier, reported separately.

---

## 6. Classification heads (identical for all representations)

| Head | Why |
|---|---|
| Logistic regression | Linear: how separable is the representation? |
| SVM (RBF kernel) | Standard strong baseline in e-nose work |
| Random forest | Nonlinear, gives feature importance |
| MLP (1 hidden layer) | Small neural head |

Same hyperparameter grid for every representation.

---

## 7. Evaluation protocol

| | GasWindTunnel | EverydayScents |
|---|---|---|
| CV | `StratifiedKFold(5)`, stratified by gas × wind speed | `StratifiedGroupKFold(5)`, groups = session |
| Per fold | 480 train / 120 test | about 124 train / 31 test |
| Repeats | 3 seeds | 3 seeds |
| Reported | mean ± std | mean ± std |

```python
# F: (N, D) features, y: (N,) labels
# GasWindTunnel: strat = gas × wind speed id, groups = None
# EverydayScents: strat = y_multi, groups = session_id
for train_idx, test_idx in cv.split(F, strat, groups):
    scaler = StandardScaler().fit(F[train_idx])
    F_tr = scaler.transform(F[train_idx])   # (N_train, D)
    F_te = scaler.transform(F[test_idx])    # (N_test, D)
    head.fit(F_tr, y[train_idx])
    f1 = f1_score(y[test_idx], head.predict(F_te), average="macro")
```

### Metrics

| Metric | Use |
|---|---|
| Macro F1 | Primary, both tasks |
| Balanced accuracy | Secondary |
| ROC AUC | Binary tasks |
| Confusion matrix | Best model per representation, multiclass |

---

## 8. Comparisons

### C1. Main result: representation × head

One table per dataset and task (4 tables). Cell value: macro F1, mean ± std.

| Representation | Logistic regression | SVM | Random forest | MLP |
|---|---|---|---|---|
| R1 Raw | | | | |
| R2 Handcrafted | | | | |
| R3 Encoder | | | | |
| Reference: end to end CNN | (single value) | | | |

### C2. Lab vs real world

Compare the ranking of R1, R2, R3 on GasWindTunnel against EverydayScents. Does the best representation change from pure gases with many samples to real scents with few samples?

### C3. Label efficiency (GasWindTunnel)

Train with 10%, 25%, 50% and 100% of each training fold; plot macro F1 against the fraction of labels, one line per representation. (EverydayScents is already in the low label regime.)

### C4. Cost

Feature size, time to build features, time to train the best head, GPU needed, for each representation and dataset.

### C5. Sanity check: baseline only

Train the same heads on the **baseline part only**, before any gas or scent is presented. A fair task scores near chance (10% for GasWindTunnel multiclass, 25% for EverydayScents). Both datasets need this check: GasWindTunnel was recorded in batches over 16 months, and in EverydayScents black pepper was only recorded on 4 days, so the resting level of the sensors could reveal the label.

### C6. Small ablations

| Representation | Ablation |
|---|---|
| R1 | T' at half and double the default |
| R2 | steady state features (1 to 4) vs all 12 |
| R3 | 16 vs 32 vs 64 features per sensor |

---

## 9. From results to the guide

A decision table filled in **from our results**:

| Your situation | Recommended representation | Evidence |
|---|---|---|
| Lab data, many labeled recordings | | C1 GasWindTunnel, C3 |
| Real world scents, few samples | | C1 EverydayScents, C2 |
| Need interpretable features | | C1, random forest importance |
| No GPU, limited time | | C4 |
| Detecting one target (binary) | | C1 binary tables |
| Identifying many classes | | C1 multiclass tables |

Plus a data setup checklist: baseline correction, split by session, train only fitting, baseline only sanity check.

---

## 10. Folder structure

Shared code lives in `src/`; each notebook calls it and stays short. Every notebook writes its scores to `results/` as a csv, so the comparison notebooks and the report read only from there.

```
gas-repr-guide/
├── README.md                          # how to run, in order
├── requirements.txt
├── data/
│   ├── raw/
│   │   ├── gas_wind_tunnel/           # UCI 251 download (not committed)
│   │   └── everyday_scents/           # session_*.csv + session_*_metadata.json
│   └── processed/
│       ├── gas_wind_tunnel/           # X_signals.npy (600, 72, 1500), meta.csv
│       └── everyday_scents/           # X_signals.npy (155, 11, 210), meta.csv
├── notebooks/
│   ├── 00_data/
│   │   ├── 00a_gwt_audit_and_subset.ipynb        # pick location + voltage, build 600 subset
│   │   ├── 00b_es_label_cleaning.ipynb           # raw labels → 4 classes, drop tests
│   │   ├── 00c_es_segmentation.ipynb             # session csv → per sample (11, 210)
│   │   └── 00d_eda_both.ipynb                    # example curves, class counts
│   ├── 01_naive_raw/
│   │   ├── 01a_gwt_raw.ipynb                     # R1 + 4 heads, both tasks
│   │   └── 01b_es_raw.ipynb
│   ├── 02_handcrafted/
│   │   ├── 02a_feature_definitions.ipynb         # each feature plotted on one curve
│   │   ├── 02b_gwt_handcrafted.ipynb             # R2 + 4 heads, both tasks
│   │   └── 02c_es_handcrafted.ipynb
│   ├── 03_encoder/
│   │   ├── 03a_gwt_encoder.ipynb                 # R3 + 4 heads, both tasks
│   │   ├── 03b_es_encoder.ipynb
│   │   └── 03c_end_to_end_reference.ipynb        # full CNN, both datasets
│   └── 04_comparisons/
│       ├── 04a_main_tables.ipynb                 # C1, C2
│       ├── 04b_label_efficiency.ipynb            # C3
│       ├── 04c_cost.ipynb                        # C4
│       ├── 04d_baseline_sanity_check.ipynb       # C5
│       ├── 04e_ablations.ipynb                   # C6
│       └── 04f_guide_table.ipynb                 # Section 9
├── src/
│   ├── load_gwt.py                    # GasWindTunnel loader
│   ├── load_es.py                     # EverydayScents loader, label map, segmentation
│   ├── preprocess.py                  # resample, baseline correction
│   ├── features/
│   │   ├── raw.py                     # R1
│   │   ├── handcrafted.py             # R2, one function per feature
│   │   └── encoder.py                 # R3 CNN + training loop
│   ├── heads.py                       # 4 heads + grids
│   └── cv.py                          # folds per dataset, metrics, writes results/
├── results/                           # one csv per notebook: seed, fold, task, rep, head, metrics
└── report/
    ├── figures/
    └── tables/
```

**Naming:** `gwt` = GasWindTunnel, `es` = EverydayScents. Notebooks run in number order.

### Suggested ownership

| Member | Owns |
|---|---|
| A | `00_data/`, `load_*.py`, `preprocess.py`, `01_naive_raw/` |
| B | `02_handcrafted/`, `heads.py`, `cv.py` |
| C | `03_encoder/` |
| All | `04_comparisons/`, report |

---

## 11. Timeline (adjust to the course deadline)

| Week | Work | Notebooks |
|---|---|---|
| 1 | Both datasets: audit, label cleaning, segmentation, processed arrays | 00a to 00d |
| 2 | CV code, heads, R1 on both datasets, baseline sanity check | 01a, 01b, 04d |
| 3 | R2 on both datasets | 02a to 02c |
| 4 | R3 and end to end reference on both datasets | 03a to 03c |
| 5 | Label efficiency, cost, ablations, main tables | 04a to 04c, 04e |
| 6 | Guide table, report, presentation | 04f |

---

## 12. Risks and fallbacks

| Risk | Fallback |
|---|---|
| GasWindTunnel raw files are very large | Load only the chosen location and voltage |
| Baseline only check scores well above chance | Keep baseline correction, report the leak clearly as a finding |
| EverydayScents folds unbalanced (black pepper has only 8 sessions) | Check class counts per fold; if needed use 4 folds or leave one day out |
| Encoder overfits on EverydayScents (about 124 training samples) | Smaller encoder, early stopping; report it as a finding for the guide |
| Every representation scores near 100% on GasWindTunnel | Rely on the label efficiency study (C3) to separate them |
| Time runs out | Drop C6 and the end to end reference; C1, C2 and C5 still answer the research question |