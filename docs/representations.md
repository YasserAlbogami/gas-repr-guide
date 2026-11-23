# The three representations

A representation turns one recording, a block of sensor curves of shape (C, T), into **one vector**. Stacking the vectors gives a feature matrix F of shape (N, D), which goes to the 4 classification heads (logistic regression, SVM, random forest, MLP).

All three start from the same preprocessed input, `X_signals.npy` (see `data/DATASET.md`):

| | GasWindTunnel (UCI) | EverydayScents (ES) |
|---|---|---|
| X_signals | (600, 72, 1500) | (144, 11, 210) |
| Baseline points | 0 – 149 | 0 – 59 |
| Exposure points | 150 – 1499 | 60 – 209 |

They are **three separate branches**. None of them uses the output of another:

```
                          ┌─ R1 raw:          downsample 1500 → 100, flatten   → (600, 7200)
X_signals (600, 72, 1500) ┼─ R2 handcrafted:  12 features per full curve      → (600, 864)
                          └─ R3 encoder:      1D CNN → 32 per full curve       → (600, 2304)

                          ┌─ R1 raw:          downsample 210 → 70, flatten     → (144, 770)
X_signals (144, 11, 210)  ┼─ R2 handcrafted:  12 features per full curve      → (144, 132)
                          └─ R3 encoder:      1D CNN → 32 per full curve       → (144, 352)
```

| | Per sensor curve | UCI F | ES F | Fitted? |
|---|---|---|---|---|
| R1 raw | 100 (UCI) / 70 (ES) averaged points | (600, 7200) | (144, 770) | no |
| R2 handcrafted | 12 hand-defined features | (600, 864) | (144, 132) | no |
| R3 encoder | 32 learned features | (600, 2304) | (144, 352) | yes, inside each fold |

---

## R1. Raw features (naive)

**Idea:** give the classifier the curve itself and let the head work out what matters.

1. **Downsample every sensor curve** by averaging blocks of consecutive points:
   - UCI: 1500 → 100 points, blocks of 15 (about 2 s each)
   - ES: 210 → 70 points, blocks of 3
2. **Flatten** the sensors one after another:
   - UCI: (72, 100) → 7200 numbers
   - ES: (11, 70) → 770 numbers

| | Per recording | F |
|---|---|---|
| UCI | (72, 1500) → (72, 100) → 7200 | **(600, 7200)** |
| ES | (11, 210) → (11, 70) → 770 | **(144, 770)** |

- **Why downsample:** the full curves would give 108,000 numbers per UCI recording for only 600 recordings. Averaging blocks keeps the shape of the curve and reduces noise.
- **The downsampling belongs to R1 only.** R2 and R3 read the full 1500 / 210 points.
- **Code:** `src/features/raw.py` (`raw_features`). It works for any T' (the C6 ablation uses half and double).

---

## R2. Hand feature engineering: 12 features per sensor

**Idea:** describe each curve with 12 standard e-nose response features, each with a clear physical meaning.

- **Input:** the full-resolution curves, 1500 points (UCI) and 210 points (ES), not R1's 100 / 70. Several features depend on time resolution (peak, time to peak, slopes, derivative). On 100 points, one UCI point would cover 2 s instead of 0.13 s.
- **Notation:**
  - `r` is the baseline-corrected curve.
  - `L` is the resting level, the mean of the raw baseline before correction (stored in `baseline_level.npy`).

### Level features

| # | Feature | Definition | What it captures |
|---|---|---|---|
| 1 | Baseline level | `L` = mean of the raw baseline | The sensor's resting level in clean air |
| 2 | Steady-state response | mean of `r` over the **last 20% of the exposure** (UCI: last 270 points, ES: last 30) | Where the sensor settles while the gas is present |
| 3 | Normalized response | steady state ÷ \|L\| | Relative change (like ΔR/R₀), less sensitive to each sensor's own offset |
| 4 | Peak response | the value of `r` with the largest absolute size during the exposure (keeps its sign) | Strongest reaction, up or down |

### Timing and shape features

| # | Feature | Definition | What it captures |
|---|---|---|---|
| 5 | Time to peak | peak index − start of exposure (in points) | How fast the sensor reaches its peak |
| 6 | Rise slope | slope of a straight line fitted from the start of the exposure to the peak | Speed of the rising response |
| 7 | Decay slope | slope of a straight line fitted from the peak to the end of the curve | Drift or decline after the peak while the gas is still present |
| 8 | Area under the curve | sum of `r` over the exposure | Total response: magnitude × duration |

### Transient features: exponential moving average (EMA) of the derivative

From the start of the exposure:

```
d[k] = r[k] − r[k−1]                     (derivative, point to point)
y[k] = (1 − α) · y[k−1] + α · d[k]       (smoothed derivative)
```

| # | Feature | Definition | What it captures |
|---|---|---|---|
| 9 | EMA max, α = 0.1 | max of `y` | Fastest rise, lightly smoothed |
| 10 | EMA min, α = 0.1 | min of `y` | Fastest fall, lightly smoothed |
| 11 | EMA max, α = 0.01 | max of `y` | Sustained rise, heavily smoothed |
| 12 | EMA min, α = 0.01 | min of `y` | Sustained fall, heavily smoothed |

### Shapes

| | Per sensor | Per recording | F |
|---|---|---|---|
| UCI | 12 | (72, 12) | **(600, 864)** |
| ES | 12 | (11, 12) | **(144, 132)** |

### Notes

- **Units are points, not seconds.**
  - Features 5–8 are counted in points: about 0.133 s per point for UCI, and 2–3 s for ES.
  - The EMA smoothing therefore covers different spans of real time in the two datasets.
  - This is fine, because the datasets are never mixed and each head is trained within one dataset. Do not compare raw R2 values across datasets.
- **The meaning of feature 7 changed when recovery was dropped.** The recovery and cleaning phases are not used, so feature 7 measures the trend after the peak *during* the exposure, not how the sensor recovers after the gas is switched off.
- **C6 ablation:** the "steady-state only" variant keeps features 1–4, giving (600, 288) and (144, 44).
- **Code:** `src/features/handcrafted.py`, with one function per feature; `handcrafted_features` builds F.

---

## R3. Encoder feature maps: 32 learned features per sensor

**Idea:** let a small 1D convolutional network (CNN) learn which curve patterns separate the classes.

### Where the 32 comes from

32 is a design choice from the plan: it is the **number of filters in the second convolutional layer**. Each filter ends up as one number per sensor curve. The C6 ablation tries 16 and 64.

### How one sensor curve becomes 32 numbers

UCI shown (ES in brackets):

```
input curve                                              (1, 1500)   [(1, 210)]
   │  Conv1d: 16 filters, kernel 7, stride 2, padding 3, then ReLU
   ▼
16 feature maps                                          (16, 750)   [(16, 105)]
   │  Conv1d: 32 filters, kernel 5, stride 2, padding 2, then ReLU
   ▼
32 feature maps                                          (32, 375)   [(32, 53)]
   │  global average pooling (mean over time)
   ▼
32 numbers                                               (32,)       [(32,)]
```

- **Conv layer 1.** Each of 16 small filters (7 weights each) slides along the curve, two points at a time. At every position it outputs how strongly that stretch of the curve matches its pattern, for example a steep rise or a bump. ReLU keeps the positive matches. The result is 16 new "curves" of half the length.
- **Conv layer 2.** Each of 32 filters looks at all 16 maps at once (16 × 5 weights) and combines them into a more complex pattern, for example "a fast rise followed by a plateau". The result is 32 maps.
- **Global average pooling.** Each of the 32 maps is averaged over time into one number. **Number j means "how much of learned pattern j appears in this curve"**, wherever in time it happens. This step makes the output size fixed (32) whatever the curve length.

The lengths follow `out = ⌊(in + 2·padding − kernel) / stride⌋ + 1`:
- UCI: 1500 → 750 → 375
- ES: 210 → 105 → 53

### From 32 per sensor to the feature matrix

The **same encoder (shared weights)** runs on every sensor curve separately, and the results are placed side by side:

| | Per recording | F |
|---|---|---|
| UCI | 72 curves × 32 = 2304 | **(600, 2304)** |
| ES | 11 curves × 32 = 352 | **(144, 352)** |

### How the weights are learned (inside each fold, training data only)

The encoder has only **2,720 weights**:
- layer 1: 16 × 7 + 16 = 128
- layer 2: 32 × 16 × 5 + 32 = 2,592

They start random and are trained separately in every fold:

1. **Scale:** each sensor's input is divided by its standard deviation, computed on the training curves.
2. **Train:**
   - A temporary linear layer (2304 → number of classes) is placed on top.
   - Encoder and linear layer are trained together to predict the class (Adam, cross-entropy with balanced class weights).
   - Training stops early when the loss on a 20% validation part of the training fold stops improving. For ES, that part is split by session.
3. **Freeze:** the linear layer is removed and the encoder weights are fixed.
4. **Encode:** the frozen encoder turns every training and test recording into its feature vector.
5. **Heads:** the 4 heads are trained on these features, exactly as on R1 and R2.

The test recordings are only passed through the frozen encoder; they never influence its weights.

### Reference point: end to end CNN

The same network, with its linear layer kept, is also used directly as a classifier (notebook 03c). It is reported separately as a reference, not as a fourth representation.

### Notes

- **R2 vs R3:**
  - R2's 12 features are formulas we chose, so each one has a name and a meaning.
  - R3's 32 are patterns the network found useful. They are often stronger, but they have no names.
- **Features differ between folds.** The encoder is retrained in every fold, so feature j in fold 1 is not the same pattern as feature j in fold 2.
- **Cost:** R3 is the only representation that needs training. It runs on CPU and is the slowest step (notebooks 03a, 03b, 04b, 04e).
- **Code:** `src/features/encoder.py`:
  - `SensorEncoder`, `train_classifier`, `encoder_builder`;
  - `run_end_to_end` for the reference.

---

## Common to all three

- **Same everything except the representation:** the same preprocessed input, folds, seeds, heads and hyperparameter grids.
- **Scaling:** a `StandardScaler` is fitted on the training fold's features (inside the head pipeline) before every head.
- **Nothing is fitted on test data.**
  - R1 and R2 fit nothing. Their features are fixed per-sample formulas, so they are computed once for all recordings.
  - R3, the scaler and the heads are fitted on the training fold only.
