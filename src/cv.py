"""Cross validation per dataset, metrics, and writing results/.

- folds: GasWindTunnel  StratifiedKFold(5), stratified by gas x wind speed (480 train / 120 test)
         EverydayScents StratifiedGroupKFold(5), groups = session, stratified by class (~115 / 29)
         3 seeds each
- run_cv: per seed and fold, using the TRAINING fold only:
    1. build the features (R3 trains its encoder here)
    2. for every head: StandardScaler + head in one Pipeline, hyperparameters chosen by
       GridSearchCV on an inner split of the training fold (grouped by session for EverydayScents)
    3. predict the test fold and score it
- metrics: macro F1 (primary), balanced accuracy, ROC AUC (binary tasks)
- results/<name>.csv: one row per dataset, task, rep, head, seed, fold
  results/<name>_preds.csv: one row per test sample (for confusion matrices in 04a)

Shapes: X (N, C, T) signals, F (N, D) features, y (N,) labels.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import (GridSearchCV, StratifiedGroupKFold, StratifiedKFold,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.heads import make_heads

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FOLDER = {"gwt": "gas_wind_tunnel", "es": "everyday_scents"}
SEEDS = [0, 1, 2]
N_FOLDS = 5
N_INNER = 3
TASKS = ["multi", "binary"]


# ------------------------------------------------------------------ data and folds

def load_processed(dataset):
    """Processed arrays of one dataset ("gwt" or "es").

    returns X (N, C, T) float32 baseline corrected, meta DataFrame (N rows), info dict
    """
    d = PROCESSED / FOLDER[dataset]
    X = np.load(d / "X_signals.npy")                                    # (N, C, T)
    meta = pd.read_csv(d / "meta.csv")
    with open(d / "info.json") as f:
        info = json.load(f)
    return X, meta, info


def load_baseline_level(dataset):
    """Resting level of every sensor before baseline correction: (N, C).
    Used by R2 (feature 1) and by the baseline only check (C5)."""
    return np.load(PROCESSED / FOLDER[dataset] / "baseline_level.npy")


def split_columns(dataset, meta):
    """strat (N,) used to stratify the outer folds, groups (N,) session ids or None."""
    if dataset == "gwt":
        return meta["strat"].to_numpy(), None
    return meta["y_multi"].to_numpy(), meta["session_id"].to_numpy()


def make_folds(dataset, meta, seed):
    """Outer folds: list of (train_idx, test_idx)."""
    strat, groups = split_columns(dataset, meta)
    dummy = np.zeros(len(strat))
    if groups is None:
        cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
        return list(cv.split(dummy, strat))
    cv = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed)
    return list(cv.split(dummy, strat, groups))


def inner_cv(dataset, seed):
    """Inner split of the training fold, used only for choosing hyperparameters."""
    if dataset == "gwt":
        return StratifiedKFold(N_INNER, shuffle=True, random_state=seed)
    return StratifiedGroupKFold(N_INNER, shuffle=True, random_state=seed)


def subsample_train(train_idx, y, fraction, seed):
    """Stratified fraction of the training indices (label efficiency study, C3).
    Only training indices are touched; the test fold stays the same."""
    if fraction >= 1:
        return train_idx
    keep, _ = train_test_split(train_idx, train_size=fraction, stratify=y[train_idx],
                               random_state=seed)
    return np.sort(keep)


# ------------------------------------------------------------------ features

def precomputed(F_tr, y_tr, F_te, groups_tr, seed):
    """build_features for representations that fit nothing (R1, R2).

    Their features are computed per sample, once, before run_cv, and passed as X = F (N, D):
    no information can flow between samples, so this is not leakage.
    """
    return F_tr, F_te


# ------------------------------------------------------------------ the fold loop

def positive_score(model, F):
    """Score of the positive class, for ROC AUC."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(F)[:, 1]
    return model.decision_function(F)


def run_cv(X, meta, dataset, task, rep, build_features, seeds=SEEDS, heads=None,
           train_fraction=1.0, verbose=True):
    """Evaluate one representation with the 4 heads on one dataset and task.

    X:              (N, C, T) signals (or any array indexed by sample on axis 0)
    task:           "multi" or "binary"
    rep:            name written to the results, e.g. "R1_raw"
    build_features: function(X_tr, y_tr, X_te, groups_tr, seed) -> F_tr (n_tr, D), F_te (n_te, D)
                    it may fit things, but only on X_tr / y_tr
    heads:          list of head names to run (default: all 4)
    train_fraction: < 1 keeps only a stratified part of each training fold (C3)

    returns scores DataFrame (one row per seed, fold, head), preds DataFrame (one row per test sample)
    """
    y = meta[f"y_{task}"].to_numpy()                                     # (N,)
    _, groups = split_columns(dataset, meta)
    binary = task == "binary"
    rows, preds = [], []
    for seed in seeds:
        for fold, (tr, te) in enumerate(make_folds(dataset, meta, seed)):
            tr = subsample_train(tr, y, train_fraction, seed)
            g_tr = None if groups is None else groups[tr]

            t0 = time.perf_counter()
            F_tr, F_te = build_features(X[tr], y[tr], X[te], g_tr, seed)   # (n_tr, D), (n_te, D)
            feat_seconds = time.perf_counter() - t0

            for name, (head, grid) in make_heads(seed).items():
                if heads is not None and name not in heads:
                    continue
                model = Pipeline([("scaler", StandardScaler()), ("head", head)])
                search = GridSearchCV(model, {f"head__{k}": v for k, v in grid.items()},
                                      scoring="f1_macro", cv=inner_cv(dataset, seed), n_jobs=-1)
                t0 = time.perf_counter()
                search.fit(F_tr, y[tr], groups=g_tr)                     # training fold only
                fit_seconds = time.perf_counter() - t0

                y_pred = search.predict(F_te)                            # (n_te,)
                rows.append({
                    "dataset": dataset, "task": task, "rep": rep, "head": name,
                    "seed": seed, "fold": fold, "train_fraction": train_fraction,
                    "n_train": len(tr), "n_features": F_tr.shape[1],
                    "macro_f1": f1_score(y[te], y_pred, average="macro"),
                    "balanced_acc": balanced_accuracy_score(y[te], y_pred),
                    "roc_auc": roc_auc_score(y[te], positive_score(search, F_te)) if binary else np.nan,
                    "best_params": json.dumps({k.replace("head__", ""): v
                                               for k, v in search.best_params_.items()}, default=str),
                    "feat_seconds": feat_seconds, "fit_seconds": fit_seconds,
                })
                preds.append(pd.DataFrame({
                    "dataset": dataset, "task": task, "rep": rep, "head": name,
                    "seed": seed, "fold": fold, "train_fraction": train_fraction,
                    "index": te, "y_true": y[te], "y_pred": y_pred}))
            if verbose:
                last = pd.DataFrame(rows[-len(heads or make_heads(seed)):])
                print(f"{dataset} {task} {rep} seed {seed} fold {fold}: "
                      + "  ".join(f"{h} {f:.3f}" for h, f in zip(last["head"], last["macro_f1"])))
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)


# ------------------------------------------------------------------ results

def save_results(scores, preds, name):
    """Write results/<name>.csv and results/<name>_preds.csv."""
    RESULTS.mkdir(exist_ok=True)
    scores.to_csv(RESULTS / f"{name}.csv", index=False)
    if preds is not None:
        preds.to_csv(RESULTS / f"{name}_preds.csv", index=False)


def summarize(scores, metric="macro_f1"):
    """Mean and std of a metric over all seeds and folds (15 values per cell).

    returns DataFrame indexed by (dataset, task, rep) with one "mean ± std" column per head
    """
    g = scores.groupby(["dataset", "task", "rep", "head"])[metric].agg(["mean", "std"])
    cell = g.apply(lambda r: f"{r['mean']:.3f} ± {r['std']:.3f}", axis=1)
    return cell.unstack("head")
