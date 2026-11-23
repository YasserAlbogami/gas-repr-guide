# Architecture

System design of the feature representation study. The source of truth for every number here is [`docs/implementation_plan.md`](docs/implementation_plan.md).

## 1. Pipeline

```
         GasWindTunnel (UCI 251)                    EverydayScents (own sessions)
  data/raw/gas_wind_tunnel/                  data/raw/everyday_scents/session_<id>.csv + _metadata.json
                 │  00a                                     │  00b labels, 00c segmentation
                 ▼                                          ▼
   load_gwt.py + preprocess.py                 load_es.py + preprocess.py
   resample → 1500, baseline correction        voltage_v, baseline→60 + sampling→150, baseline correction
                 │                                          │
                 ▼                                          ▼
   X_signals.npy (600, 72, 1500)               X_signals.npy (144, 11, 210)
   meta.csv (600 rows)                         meta.csv (144 rows, with session_id)
                 └──────────────────┬───────────────────────┘
                                    ▼
             src/features/: raw.py (R1) | handcrafted.py (R2) | encoder.py (R3)
                                    ▼
                F (N, D) ──► src/cv.py fold loop ──► src/heads.py (4 heads)
                                    ▼
                       results/<notebook>.csv
                                    ▼
                  04_comparisons ──► report/tables, report/figures
```

The two datasets have different channels and are never merged; the same code runs on each.

## 2. Data contracts

| Artifact | GasWindTunnel | EverydayScents |
|---|---|---|
| `X_signals.npy` | (600, 72, 1500), baseline corrected, not scaled | (144, 11, 210), baseline corrected, not scaled (11 incomplete recordings dropped) |
| `y_multi` | (600,), 10 gases, 60 each | (144,), 4 classes, 33 / 39 / 50 / 22 |
| `y_binary` | (600,), CO vs other, 60 / 540 | (144,), truffle vs not, 50 / 94 |
| Grouping | strat = gas × wind speed | groups = session_id, strat = y_multi |

Every results csv has the columns: seed, fold, task, rep, head, metrics.

## 3. Representations

| | Operation | GasWindTunnel F | EverydayScents F | Fitted? |
|---|---|---|---|---|
| R1 raw | downsample to T' (100 / 70), flatten | (600, 7200) | (144, 770) | no |
| R2 handcrafted | 12 features per sensor | (600, 864) | (144, 132) | no |
| R3 encoder | shared 1D CNN per sensor, 32 channels after GAP | (600, 2304) | (144, 352) | yes, per fold |

R3 training per fold: encoder + temporary linear layer trained on training labels, linear layer removed, encoder frozen, features extracted for train and test. The end to end CNN (03c) is a separate reference point.

## 4. Evaluation and leakage control

- GasWindTunnel: `StratifiedKFold(5)`, 480 / 120 per fold. EverydayScents: `StratifiedGroupKFold(5)` by session, about 115 / 29. 3 seeds each, mean ± std.
- Inside each fold: StandardScaler, encoder and heads are fitted on the training part only. Hyperparameters come from an inner validation split of the training part.
- Metrics: macro F1 (primary), balanced accuracy, ROC AUC (binary), confusion matrix (best model per representation).
- Sanity check C5: heads fitted on the baseline segment only must score near chance (10% / 25%).
