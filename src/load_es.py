"""EverydayScents loader: label map and segmentation of the session files.

Files in data/raw/everyday_scents/:
    session_<id>.csv            long format: timestamp, phase, sensor, value, raw, voltage_v, gain
    session_<id>_metadata.json  samples (label, phase durations, gains, phase_history)

One session holds one or more samples. Every sample's phase_history lists all phases of the session
up to that sample, so the sample's own baseline and sampling windows are the LAST
BASELINE_MEASUREMENT and SAMPLE_MEASUREMENT entries of its list.

Only the 11 gas channels are used, through voltage_v (independent of the gain setting);
gas_resistance_ohm, temperature_c, humidity_percent and pressure_hpa are not used as features.
"""
import glob
import json
import os

import numpy as np
import pandas as pd

CHANNELS = ["hcho", "ch4", "voc", "odor", "etoh", "h2s", "no2", "nh3", "co", "smoke", "h2"]
CLASSES = ["ambient", "coffee", "truffle", "black_pepper"]
TRUFFLE_CLASS = CLASSES.index("truffle")      # positive class of the binary task

# raw label -> class (Section 2 of the plan); labels not listed here are dropped
LABEL_MAP = {
    "ambient": "ambient",
    "coffee": "coffee", "Arabica_coffee": "coffee", "Robusta_coffee": "coffee",
    "coffee_outdoor": "coffee", "coffee ground": "coffee", "coffee_ground_heated": "coffee",
    "truffle": "truffle", "fresh_truffle": "truffle", "fresh_truffle_with_jar": "truffle",
    "truffle_outdoor": "truffle",
    "black_pepper": "black_pepper", "black_paper": "black_pepper",   # black_paper is a typo
}


def _last_phase(history, name):
    """(start, end) timestamps of the last phase called name in a phase_history list."""
    entries = [p for p in history if p["phase"] == name]
    if not entries:
        return None, None
    p = entries[-1]
    return pd.Timestamp(p["start"]), (pd.Timestamp(p["end"]) if p["end"] else None)


def load_samples(raw_dir):
    """One row per sample of every session (all labels, nothing dropped yet).

    returns DataFrame: session_id, sample_id, day, raw_label, class (None if dropped),
                       base_start, base_end, samp_start, samp_end
    """
    rows = []
    for path in sorted(glob.glob(os.path.join(raw_dir, "session_*_metadata.json"))):
        with open(path, encoding="utf-8") as f:
            meta = json.load(f)
        for s in meta["samples"]:
            base_start, base_end = _last_phase(s["phase_history"], "BASELINE_MEASUREMENT")
            samp_start, samp_end = _last_phase(s["phase_history"], "SAMPLE_MEASUREMENT")
            rows.append({
                "session_id": meta["session_id"],
                "sample_id": s["sample_id"],
                "day": meta["session_id"].split("_")[1],          # YYYYMMDD
                "raw_label": s["label"],
                "class": LABEL_MAP.get(s["label"]),
                "base_start": base_start, "base_end": base_end,
                "samp_start": samp_start, "samp_end": samp_end,
            })
    return pd.DataFrame(rows)


def read_session_csv(raw_dir, session_id):
    """The session csv with parsed timestamps, gas channels only.

    returns DataFrame: timestamp, phase, sensor, voltage_v
    """
    df = pd.read_csv(os.path.join(raw_dir, f"{session_id}.csv"))
    df = df[df["sensor"].isin(CHANNELS)]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df[["timestamp", "phase", "sensor", "voltage_v"]]


def _window(df, phase, start, end):
    """Rows of one phase between start and end, pivoted to wide.

    returns t (T_raw,) seconds since start, x (11, T_raw) voltage_v
    """
    part = df[(df["phase"] == phase) & (df["timestamp"] >= start)]
    if end is not None:
        part = part[part["timestamp"] <= end]
    # readings of the 11 channels share (almost) one time stamp per cycle: round to 0.1 s
    part = part.assign(t=(part["timestamp"] - start).dt.total_seconds().round(1))
    wide = part.pivot_table(index="t", columns="sensor", values="voltage_v", aggfunc="mean")
    wide = wide.reindex(columns=CHANNELS).interpolate(limit_direction="both")   # (T_raw, 11)
    return wide.index.to_numpy(), wide.to_numpy().T                             # (T_raw,), (11, T_raw)


def segment(session_df, sample):
    """Cut one sample out of its session.

    session_df: output of read_session_csv for the sample's session
    sample:     one row of load_samples
    returns (t_base (Tb_raw,), x_base (11, Tb_raw)), (t_samp (Ts_raw,), x_samp (11, Ts_raw))
    """
    base = _window(session_df, "BASELINE_MEASUREMENT", sample["base_start"], sample["base_end"])
    samp = _window(session_df, "SAMPLE_MEASUREMENT", sample["samp_start"], sample["samp_end"])
    return base, samp
