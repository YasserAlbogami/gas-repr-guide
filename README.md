# Raw, Engineered, or Learned? A Feature Representation Guide for Gas Sensor Channels

A controlled comparison of three ways to turn electronic nose sensor curves into feature vectors (raw, handcrafted, learned by a 1D CNN encoder), each fed to the same classification heads with the same folds and preprocessing.
We run it on a large lab dataset of pure gases and a small real world dataset of everyday scents.
The outcome is a practical guide: which representation to choose for a gas or scent detection classifier, depending on data size, task and resources.

![Python](https://img.shields.io/badge/python-3.10+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-F7931E?logo=scikitlearn&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-latest-EE4C2C?logo=pytorch&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-notebooks-F37626?logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/status-in%20progress-yellow)

---

## 🔍 The Problem

An electronic nose has several sensors; each one gives one voltage curve over time when a gas or scent is presented. Together the curves form a fingerprint, and a classifier maps that fingerprint to a label such as "coffee" or "acetone".

Before any classifier can be trained, the C × T block of curves has to become one feature vector. This project asks which way of building that vector works best, and when:

1. **R1 Raw features (naive):** downsample the curves and flatten them.
2. **R2 Hand feature engineering:** 12 standard response features per sensor.
3. **R3 Encoder feature maps:** a small 1D CNN learns the features.

Full plan: [`docs/implementation_plan.md`](docs/implementation_plan.md).

---

## 🌟 Features

- **Two contrasting datasets**, never merged, same pipeline run on each:

| | GasWindTunnel | EverydayScents |
|---|---|---|
| Source | UCI 251, *Gas Sensor Arrays in Open Sampling Settings* (lab) | Our own, collected with a low cost e-nose (field and lab) |
| Recordings used | 600 (one board location, one heater voltage, 3 wind speeds) | 155 samples from 81 sessions |
| Channels | 72 metal oxide sensors | 11 gas channels |
| X_signals | (600, 72, 1500) | (155, 11, 210) |
| Multiclass task | 10 gases | 4 scents: ambient, coffee, truffle, black_pepper |
| Binary task | CO vs other gases | truffle vs not truffle |
| Folds | StratifiedKFold(5), stratified by gas × wind speed | StratifiedGroupKFold(5), grouped by session |

- **Identical heads** on every representation: logistic regression, SVM (RBF), random forest, MLP.
- **Leakage controls:** split by session, every scaler / encoder / head fitted on training folds only, and a baseline only sanity check that should score near chance.
- **Studies:** main tables (C1), lab vs real world (C2), label efficiency (C3), cost (C4), baseline sanity check (C5), ablations (C6).

---

## 🏗️ Architecture

```
data/raw/  ──►  00_data  ──►  data/processed/<dataset>/X_signals.npy  (N, C, T) + meta.csv
                                          │
             ┌────────────────────────────┼────────────────────────────┐
             ▼                            ▼                            ▼
       R1 raw (01_*)             R2 handcrafted (02_*)          R3 encoder (03_*)
       (N, C·T')                 (N, C·12)                      (N, C·32)
             └────────────────────────────┼────────────────────────────┘
                                          ▼
                 src/cv.py: per fold, scaler + 4 heads fitted on train only
                                          ▼
                       results/<notebook>.csv  (seed, fold, task, rep, head, metrics)
                                          ▼
                 04_comparisons  ──►  report/tables, report/figures, guide table
```

See [`Architecture.md`](Architecture.md) for details.

---

## 🚀 Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| Arrays and tables | NumPy, pandas | Signals, metadata, results csv files |
| Signal processing | SciPy | Resampling, slopes |
| Heads and CV | scikit-learn | Logistic regression, SVM, random forest, MLP, folds, metrics |
| Encoder | PyTorch | 1D CNN encoder and end to end reference |
| Plots | Matplotlib | Figures for the report |
| Notebooks | Jupyter | One notebook per step |

---

## 📥 Setup & Installation

```bash
python -m venv venv
venv\Scripts\activate          # Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
```

Place the data (not committed):

- `data/raw/gas_wind_tunnel/`: UCI 251 download
- `data/raw/everyday_scents/`: `session_<id>.csv` and `session_<id>_metadata.json`

---

## 🎯 Usage Workflow

Run the notebooks in number order (`gwt` = GasWindTunnel, `es` = EverydayScents):

1. **Data:** `00a_gwt_audit_and_subset` → `00b_es_label_cleaning` → `00c_es_segmentation` → `00d_eda_both`
2. **R1 raw:** `01a_gwt_raw` → `01b_es_raw`
3. **R2 handcrafted:** `02a_feature_definitions` → `02b_gwt_handcrafted` → `02c_es_handcrafted`
4. **R3 encoder:** `03a_gwt_encoder` → `03b_es_encoder` → `03c_end_to_end_reference`
5. **Comparisons:** `04a_main_tables` → `04b_label_efficiency` → `04c_cost` → `04d_baseline_sanity_check` → `04e_ablations` → `04f_guide_table`

Notebooks 01 to 03 write one csv each to `results/`; the 04 notebooks and the report read only from there.

---

## 📁 Folder Structure

```
.
├── README.md
├── Architecture.md
├── requirements.txt
├── docs/implementation_plan.md     # the project plan
├── data/
│   ├── raw/                        # gas_wind_tunnel/, everyday_scents/ (not committed)
│   └── processed/                  # X_signals.npy + meta.csv per dataset (not committed)
├── notebooks/
│   ├── 00_data/                    # audit, label cleaning, segmentation, EDA
│   ├── 01_naive_raw/               # R1
│   ├── 02_handcrafted/             # R2
│   ├── 03_encoder/                 # R3 + end to end reference
│   └── 04_comparisons/             # C1 to C6, guide table
├── src/
│   ├── load_gwt.py                 # GasWindTunnel loader
│   ├── load_es.py                  # EverydayScents loader, label map, segmentation
│   ├── preprocess.py               # resample, baseline correction
│   ├── features/                   # raw.py, handcrafted.py, encoder.py
│   ├── heads.py                    # 4 heads + grids
│   └── cv.py                       # folds, metrics, writes results/
├── results/                        # one csv per notebook
└── report/                         # figures/, tables/
```

---

## 👥 Team

| Member | Owns |
|---|---|
| A | `00_data/`, `load_*.py`, `preprocess.py`, `01_naive_raw/` |
| B | `02_handcrafted/`, `heads.py`, `cv.py` |
| C | `03_encoder/` |
| All | `04_comparisons/`, report |
