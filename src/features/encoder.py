"""R3, encoder feature maps: a small 1D CNN with shared weights over sensors, and its training loop.

Architecture, applied to each sensor curve separately (shapes for one sensor, GWT / ES):
  input                                            (1, 1500)    / (1, 210)
  Conv1d(1 -> 16, k 7, stride 2, pad 3) + ReLU     (16, 750)    / (16, 105)
  Conv1d(16 -> n_out, k 5, stride 2, pad 2) + ReLU (n_out, 375) / (n_out, 53)
  global average pooling                           (n_out,)
All sensors flattened: (N, C * n_out); with n_out = 32: GWT (600, 2304), ES (144, 352).

Training inside a fold, on the training fold only:
  1. per-sensor scale (std of the training curves), so that sensors with large readings do not dominate
  2. encoder + temporary linear layer trained on the task labels (Adam, cross entropy with balanced
     class weights), early stopping on an inner validation split (20%, grouped by session for ES)
  3. the linear layer is dropped, the encoder is frozen and encodes the train and test curves

The end to end reference (03c) is the same network used directly as the classifier.
"""
import copy
import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from torch import nn

from src.cv import make_folds, split_columns

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SensorEncoder(nn.Module):
    """Shared 1D CNN applied to every sensor curve. (B, C, T) -> (B, C * n_out)."""

    def __init__(self, n_out=32):
        super().__init__()
        self.n_out = n_out
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3), nn.ReLU(),
            nn.Conv1d(16, n_out, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x):                                   # x: (B, C, T)
        B, C, T = x.shape
        z = self.net(x.reshape(B * C, 1, T))                # (B * C, n_out, 1)
        return z.reshape(B, C * self.n_out)                 # (B, C * n_out)


class EncoderClassifier(nn.Module):
    """Encoder + linear layer. Holds the per-sensor scale fitted on the training fold."""

    def __init__(self, n_channels, n_classes, n_out=32):
        super().__init__()
        self.encoder = SensorEncoder(n_out)
        self.linear = nn.Linear(n_channels * n_out, n_classes)
        self.register_buffer("scale", torch.ones(1, n_channels, 1))

    def features(self, x):                                  # x: (B, C, T) -> (B, C * n_out)
        return self.encoder(x / self.scale)

    def forward(self, x):                                   # x: (B, C, T) -> (B, n_classes)
        return self.linear(self.features(x))


def _validation_split(y, groups, seed):
    """20% of the training fold for early stopping (grouped by session if groups is given)."""
    dummy = np.zeros(len(y))
    if groups is None:
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        return next(cv.split(dummy, y))
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    return next(cv.split(dummy, y, groups))


def _batches(n, batch_size, rng):
    order = rng.permutation(n)
    return [order[i:i + batch_size] for i in range(0, n, batch_size)]


def train_classifier(X_tr, y_tr, groups_tr, seed, n_out=32, max_epochs=60, patience=8,
                     lr=1e-3, batch_size=32, verbose=False):
    """Train encoder + linear layer on one training fold.

    X_tr: (n, C, T), y_tr: (n,) labels 0..K-1, groups_tr: (n,) sessions or None
    returns the trained EncoderClassifier (best validation loss), in eval mode
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n_classes = int(y_tr.max()) + 1
    fit_idx, val_idx = _validation_split(y_tr, groups_tr, seed)

    model = EncoderClassifier(X_tr.shape[1], n_classes, n_out).to(DEVICE)
    scale = X_tr[fit_idx].std(axis=(0, 2)) + 1e-8                    # (C,) from training curves only
    model.scale.copy_(torch.tensor(scale, dtype=torch.float32).reshape(1, -1, 1))

    counts = np.bincount(y_tr[fit_idx], minlength=n_classes)
    weight = torch.tensor(len(fit_idx) / (n_classes * np.maximum(counts, 1)), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=weight.to(DEVICE))
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    Xf = torch.tensor(X_tr[fit_idx], dtype=torch.float32)            # (n_fit, C, T)
    yf = torch.tensor(y_tr[fit_idx], dtype=torch.long)               # (n_fit,)
    Xv = torch.tensor(X_tr[val_idx], dtype=torch.float32).to(DEVICE)  # (n_val, C, T)
    yv = torch.tensor(y_tr[val_idx], dtype=torch.long).to(DEVICE)    # (n_val,)

    best_loss, best_state, bad_epochs = np.inf, None, 0
    for epoch in range(max_epochs):
        model.train()
        for b in _batches(len(Xf), batch_size, rng):
            opt.zero_grad()
            loss = loss_fn(model(Xf[b].to(DEVICE)), yf[b].to(DEVICE))
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv), yv).item()
        if verbose:
            print(f"  epoch {epoch:3d}  val loss {val_loss:.4f}")
        if val_loss < best_loss - 1e-4:
            best_loss, best_state, bad_epochs = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def encode(model, X, batch_size=64):
    """Frozen encoder features. X: (n, C, T) -> (n, C * n_out) numpy."""
    out = [model.features(torch.tensor(X[i:i + batch_size], dtype=torch.float32).to(DEVICE)).cpu()
           for i in range(0, len(X), batch_size)]
    return torch.cat(out).numpy()


@torch.no_grad()
def predict_proba(model, X, batch_size=64):
    """Class probabilities of the end to end network. X: (n, C, T) -> (n, K) numpy."""
    out = [torch.softmax(model(torch.tensor(X[i:i + batch_size], dtype=torch.float32).to(DEVICE)), 1).cpu()
           for i in range(0, len(X), batch_size)]
    return torch.cat(out).numpy()


def encoder_builder(n_out=32, **train_kwargs):
    """build_features for cv.run_cv: train the encoder on the training fold, then encode both parts."""
    def build(X_tr, y_tr, X_te, groups_tr, seed):
        model = train_classifier(X_tr, y_tr, groups_tr, seed, n_out=n_out, **train_kwargs)
        return encode(model, X_tr), encode(model, X_te)              # (n_tr, C*n_out), (n_te, C*n_out)
    return build


def run_end_to_end(X, meta, dataset, task, seeds=(0, 1, 2), n_out=32, verbose=True, **train_kwargs):
    """Reference point: the CNN used directly as the classifier, same folds as run_cv.

    returns scores DataFrame (head = "end_to_end"), preds DataFrame
    """
    y = meta[f"y_{task}"].to_numpy()                                 # (N,)
    _, groups = split_columns(dataset, meta)
    rows, preds = [], []
    for seed in seeds:
        for fold, (tr, te) in enumerate(make_folds(dataset, meta, seed)):
            g_tr = None if groups is None else groups[tr]
            t0 = time.perf_counter()
            model = train_classifier(X[tr], y[tr], g_tr, seed, n_out=n_out, **train_kwargs)
            fit_seconds = time.perf_counter() - t0
            proba = predict_proba(model, X[te])                      # (n_te, K)
            y_pred = proba.argmax(axis=1)                            # (n_te,)
            rows.append({
                "dataset": dataset, "task": task, "rep": "R3_end_to_end", "head": "end_to_end",
                "seed": seed, "fold": fold, "train_fraction": 1.0,
                "n_train": len(tr), "n_features": X.shape[1] * n_out,
                "macro_f1": f1_score(y[te], y_pred, average="macro"),
                "balanced_acc": balanced_accuracy_score(y[te], y_pred),
                "roc_auc": roc_auc_score(y[te], proba[:, 1]) if task == "binary" else np.nan,
                "best_params": json.dumps({"n_out": n_out}),
                "feat_seconds": 0.0, "fit_seconds": fit_seconds,
            })
            preds.append(pd.DataFrame({
                "dataset": dataset, "task": task, "rep": "R3_end_to_end", "head": "end_to_end",
                "seed": seed, "fold": fold, "train_fraction": 1.0,
                "index": te, "y_true": y[te], "y_pred": y_pred}))
            if verbose:
                print(f"{dataset} {task} end_to_end seed {seed} fold {fold}: {rows[-1]['macro_f1']:.3f}")
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)
