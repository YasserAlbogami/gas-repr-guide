# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Course project (CS220): compare three feature representations of e-nose sensor curves (R1 raw, R2 handcrafted, R3 1D CNN encoder) under identical heads, folds and preprocessing, on two datasets, and turn the results into a representation guide. `Architecture.md` has the system design.

## Rules

- **Follow `docs/implementation_plan.md`.** It fixes the datasets, shapes, features, CNN layers, heads, CV scheme and comparisons. Do not change these without being asked; if something in the plan cannot work, say so instead of silently deviating.
- **Keep code simple and explainable.** This is a course project that the team must be able to present line by line: plain functions, NumPy / scikit-learn / PyTorch only, no frameworks or clever abstractions.
- **Define array shapes in comments**, e.g. `# X: (N, C, T)`, `# F: (N, D)`, on every array that is created or reshaped.
- **Never fit anything on test data.** Scalers, encoders, heads and hyperparameter searches are fitted on the training fold only (inner validation split for tuning). EverydayScents is always split by session (`StratifiedGroupKFold`, groups = session_id).

## Structure that spans files

- Shared code in `src/`; notebooks stay short and call it. Notebooks run in number order (`00a` … `04f`). `gwt` = GasWindTunnel, `es` = EverydayScents.
- `00_data` notebooks write `data/processed/<dataset>/X_signals.npy` (baseline corrected, **not** scaled) + `meta.csv`. Expected: GWT (600, 72, 1500), ES (155, 11, 210).
- Notebooks 01–03 each write one csv to `results/` (columns: seed, fold, task, rep, head, metrics). The `04_comparisons` notebooks and the report read only from `results/`, never recompute scores.
- The two datasets are never merged (72 vs 11 channels); the same functions run on each.
- `data/raw/`, `data/processed/` and model checkpoints are git-ignored.

## Commands

```bash
pip install -r requirements.txt
jupyter notebook                                          # run notebooks in number order
jupyter nbconvert --to notebook --execute --inplace notebooks/01_naive_raw/01a_gwt_raw.ipynb   # run one notebook headless
```

No test suite or linter is set up yet.
