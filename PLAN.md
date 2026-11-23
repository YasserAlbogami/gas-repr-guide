# Plan: implement the data stage (and run it), write the code for R1, R2, R3 and the comparisons (not run)

## Context

The repo is scaffolded: 18 notebooks with only a title and a description cell, and `src/` modules with docstrings only. You want to start the experiments. In this round I will:
- get both datasets onto disk;
- implement and **run** the `00_data` notebooks, so the processed arrays are ready for all three representations;
- write the shared `src/` code and the notebooks for R1 raw, R2 handcrafted, R3 encoder and the comparisons, **without running them**. You then run 01 → 02 → 03 → 04 yourself, step by step.

Everything follows `docs/implementation_plan.md` and the `CLAUDE.md` rules: simple code, array shapes in comments, nothing fitted on test data.

### What the data looks like (checked read-only)

**GasWindTunnel (UCI 251)**
- The zip has paths like `WTD_upload/<Gas>_<ppm>/L<1-6>/<timestamp>_board_setPoint_<400-600>V_fan_setPoint_<000|060|100>_mfc_setPoint_<Gas>_<ppm>ppm_p<k>`.
- Each file is a text table with 92 columns. Column 0 is time in ms (0 to 260 s, about 50 to 100 Hz, uneven steps). Columns 1 to 8 are fan and gas-flow (MFC) set points; 00a will confirm which column marks the gas release. Columns 9 and 10 are temperature and humidity. Then come 9 boards × (1 flag + 8 sensors), giving 72 sensor columns.
- The UCI server **ignores HTTP range requests**, so a download cannot resume; it always restarts from zero. The full zip is roughly 8 GB (estimated from about 450 KB per file × 18,000 files). Unzipping all of it would take about 100 GB, so **we never unzip everything**: 00a reads the file names from the zip and extracts only the 600 recordings it needs.

**EverydayScents**
- Already extracted in `data/raw/everyday_scents/`: 81 sessions and 172 samples. With the plan's label map this gives exactly **155 usable samples** (37 ambient / 42 coffee / 54 truffle / 22 black_pepper), and 17 samples are dropped.
- The CSV has no sample_id column. Each sample's `phase_history` in the metadata gives the start and end times of its own `BASELINE_MEASUREMENT` and `SAMPLE_MEASUREMENT` phases (its last occurrence in the list), so segmentation means cutting the CSV by timestamp window.

---

## Phase 1: Data (implement + RUN)

- [x] 1.1 Fix `get_data.py`. GasWindTunnel: download the zip only (no full unzip), and drop the "resume" claim because the server doesn't support it. Make the download skip if the zip is already complete. Update the matching line in README.md.
- [x] 1.2 Download the full UCI 251 zip with `python get_data.py`, in the background (about 8 GB). EverydayScents is already in place.
- [x] 1.3 `src/preprocess.py`: `resample(x, T)` (linear interpolation on a uniform grid, `(C, T_raw) → (C, T)`) and `baseline_correct(x, n_base)` (subtract the per-sensor mean of the first `n_base` points).
- [x] 1.4 `src/load_gwt.py`:
  - `list_recordings(zip_path) → DataFrame` (gas, ppm, location, voltage, fan, timestamp, path), parsed from the file names only;
  - `read_recording(zip_path, path) → t (T_raw,), X (72, T_raw), release info`;
  - `GASES`, `CO_CLASS`.
- [x] 1.5 `src/load_es.py`:
  - `LABEL_MAP` (raw label → class, as in Section 2 of the plan);
  - `load_samples(dir) → DataFrame` (session_id, sample_id, day, raw_label, class, baseline and sampling start/end);
  - `segment(csv_df, sample) → baseline (11, Tb_raw), sampling (11, Ts_raw)`, using `voltage_v` for the 11 gas channels.
- [x] 1.6 **00a_gwt_audit_and_subset.** Do the following:
  - Count recordings per (location, voltage, gas, fan).
  - Pick the subset: complete first (10 gases × 3 fans × ≥ 20 repetitions). Ties are broken by 500 V (the middle heater voltage), then L3 (the middle distance). If CO appears at two concentrations, use the complete one, preferring the lower ppm.
  - Keep the first 20 repetitions by timestamp.
  - Find the gas release time from the flow column and plot one recording.
  - Resample each recording over 0–260 s to (72, 1500), then baseline-correct with the pre-release mean.
  - Save `X_signals.npy (600, 72, 1500)` and `meta.csv` (gas, y_multi, y_binary = CO, fan, wind_id, strat = gas × fan, rep, file). Also save `n_base` (the number of pre-release points at T = 1500) for R2 and C5.
- [x] 1.7 **00b_es_label_cleaning.** Apply the label map and check that the counts are 37/42/54/22 = 155 with 17 dropped. Show tables of class × session and class × day. Report the open item: check whether `old_dataset.csv` and `new_truffle_dataset.csv` overlap with the sessions (by timestamps); report only, the plan uses sessions only. Write `meta.csv` (155 rows: session_id, sample_id, day, raw_label, class, y_multi, y_binary).
- [x] 1.8 **00c_es_segmentation.** Do the following:
  - Cut each sample, pivot to wide, and resample baseline → 60 and sampling → 150, giving (11, 210).
  - Subtract the baseline mean per sensor.
  - Assert there are no NaN values and that every sample has enough points in both phases; flag and report any sample that doesn't.
  - Save `X_signals.npy (155, 11, 210)`. `n_base` = 60.
- [x] 1.9 **00d_eda_both.** Plot example curves per class (both datasets), class counts, and the per-fold class counts of the planned CV splits. That last plot is risk check #3 in the plan (black_pepper has only 8 sessions). Save the figures to `report/figures/`.
- [x] 1.10 Run 00a → 00d with `jupyter nbconvert --execute --inplace` so the outputs are kept in the notebooks.

**Exit check:**
- `data/processed/gas_wind_tunnel/X_signals.npy` has shape (600, 72, 1500) with 60 recordings per gas.
- `data/processed/everyday_scents/X_signals.npy` has shape (155, 11, 210) with counts 37/42/54/22.
- There are no NaN values, and all 4 notebooks run without errors.

## Phase 2: Shared experiment code + R1 naive raw (write, do NOT run)

- [x] 2.1 `src/heads.py`: `make_heads(seed)` returns 4 sklearn heads, each with a small grid:
  - LogReg: C ∈ {0.01, 0.1, 1, 10}
  - SVM-RBF: C ∈ {1, 10, 100}, gamma ∈ {scale, 0.01}
  - RF: n_estimators 300, max_depth ∈ {None, 10}
  - MLP: hidden ∈ {64, 128}, alpha ∈ {1e-4, 1e-2}
- [x] 2.2 `src/cv.py` defines `run_cv(X, y, strat, groups, build_features, heads, seeds, task, rep, dataset)`. Per seed and fold it does the following, using the training fold only:
  - `F_tr, F_te = build_features(X_tr, y_tr, X_te, groups_tr)`; this is where R3 trains its encoder.
  - Fit the scaler on `F_tr`.
  - `GridSearchCV` on an inner split: StratifiedKFold(3) for GWT, StratifiedGroupKFold(3) by session for ES.
  - Predict on the test fold.
  - Record macro F1, balanced accuracy, ROC AUC (binary tasks), and fit time.
  - Also write the per-sample predictions so 04a can draw confusion matrices without refitting.
  - Helpers: `load_processed(dataset)`, `make_folds(dataset, seed)`, `save_results(df, name)`. Outputs go to `results/<notebook>.csv` and `results/<notebook>_preds.csv`.
- [x] 2.3 `src/features/raw.py`: `raw_features(X, t_prime) → (N, C·T')`, by block-averaging downsampling.
- [x] 2.4 Notebooks **01a_gwt_raw** (T' = 100 → (600, 7200)) and **01b_es_raw** (T' = 70 → (155, 770)): load, build features, `run_cv` for both tasks with 3 seeds, then save results and a short summary table.

**Exit check:** the notebooks are complete but not run. The `src` functions are checked only on a tiny random array (shapes only, no real-data fitting).

## Phase 3: R2 handcrafted (write, do NOT run)

- [x] 3.1 `src/features/handcrafted.py`: one function per feature (1–12, as in Section 5 of the plan). They work on one curve `(T,)` plus `n_base`. `handcrafted_features(X, n_base, subset=None) → (N, C·12)` (or `(N, C·4)` for the C6 steady-state subset).
- [x] 3.2 **02a_feature_definitions**: each feature drawn on one GWT curve and one ES curve, saved to `report/figures/`.
- [x] 3.3 **02b_gwt_handcrafted** ((600, 864)) and **02c_es_handcrafted** ((155, 132)), with the same `run_cv` calls as 01.

**Exit check:** the notebooks are complete and not run, and a shape test on random input passes.

## Phase 4: R3 encoder (write, do NOT run)

- [x] 4.1 `src/features/encoder.py`:
  - `SensorEncoder(n_out=32)`: Conv1d(1→16, k7, s2, p3) + ReLU → Conv1d(16→32·, k5, s2, p2) + ReLU → global average pooling. It runs on `(N·C, 1, T)` and reshapes the result to `(N, C·n_out)`.
  - `train_encoder(X_tr, y_tr, groups_tr)`: encoder plus a temporary linear layer, Adam, with early stopping on an inner validation split (20%, grouped by session for ES). It is trained on that task's labels.
  - `encoder_builder(...)`: plugs into `run_cv` as `build_features` (train on the fold, freeze, encode train and test).
  - `EndToEndCNN` with a `run_end_to_end` loop for 03c.
  - Seeds are fixed, and it runs on CPU if there is no GPU.
- [x] 4.2 **03a_gwt_encoder** ((600, 2304)), **03b_es_encoder** ((155, 352)) and **03c_end_to_end_reference** (both datasets and both tasks, same folds).

**Exit check:** the notebooks are complete and not run, and a forward-pass shape test on random input passes.

## Phase 5: Comparisons (write, do NOT run)

- [x] 5.1 **04a_main_tables**: C1 is 4 tables (representation × head, macro F1 mean ± std, plus a row for the end-to-end reference), read from `results/` only. It also draws confusion matrices from the `_preds` files. C2 compares how the three representations rank on each dataset.
- [x] 5.2 **04b_label_efficiency**: C3 on GWT, using 10/25/50/100% of each training fold (a stratified subsample of the training fold only). Writes `results/04b_label_efficiency.csv` and a plot.
- [x] 5.3 **04c_cost**: C4 covers feature size, time to build features, time to fit the best head, and whether a GPU is used.
- [x] 5.4 **04d_baseline_sanity_check**: C5 trains the heads on baseline-only inputs, GWT `X[:, :, :n_base]` and ES `X[:, :, :60]`, using R1 and R2-baseline features. The expected result is near chance.
- [x] 5.5 **04e_ablations**: C6 varies R1 T' (half and double), R2 (features 1–4 vs all 12), and R3 (16/32/64 outputs per sensor).
- [x] 5.6 **04f_guide_table**: fills in the Section 9 decision table from the `results/` csv files, plus the checklist.

**Exit check:** all of 04 is written and not run. Every 04 notebook reads only from `results/`, except the ones that run their own extra experiments (04b–04e), and those write their own csv files.

---

## Critical files

- `get_data.py`, `README.md` (data step text)
- `src/preprocess.py`, `src/load_gwt.py`, `src/load_es.py`, `src/cv.py`, `src/heads.py`, `src/features/{raw,handcrafted,encoder}.py`
- All 18 notebooks under `notebooks/`. Each stays short and calls `src/`, adding `sys.path.append("../..")` at the top.

## Verification

1. **Data:** the asserts inside 00a–00d (shapes, class counts, no NaN), plus all 4 notebooks executed without errors via nbconvert.
2. **Code for 01–04:** a throwaway smoke script in the scratchpad that imports every `src` module and runs each feature builder and the encoder forward pass on a tiny random array to check the shapes from the plan. No real-data experiments, no notebook runs.
3. **Leakage review:** read `run_cv` once more to confirm the scaler, encoder, grid search and label-efficiency subsample all use the training indices only.

## Notes

- On approval I copy this plan to `PLAN.md` in the project root, create the todo list, and tick tasks as they finish.
- No commits or pushes unless you ask.

## Outcome (2026-11-22)

- Phase 1 run: GWT `X_signals.npy` (600, 72, 1500), L3 / 500 V, CO_4000 (CO_1000 is incomplete), n_base = 116, exposure ends at point 1155.
  ES `X_signals.npy` (144, 11, 210): 11 of the 155 labelled samples are aborted recordings (sampling phase 0-15 s) and were dropped in 00c.
  `old_dataset.csv` / `new_truffle_dataset.csv` do not overlap with the sessions (June vs July-August).
- Phases 2-5 written, not run; shape smoke test on random arrays passed.
- Environment moved to uv (`pyproject.toml` + `uv.lock`, `.venv/`).
