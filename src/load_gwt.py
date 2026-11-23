"""GasWindTunnel loader (UCI 251, Gas Sensor Arrays in Open Sampling Settings).

The files are read straight from the downloaded zip (data/raw/gas_wind_tunnel/uci251.zip);
the zip is never fully unzipped.

Zip layout:
    WTD_upload/<Gas>_<ppm>/L<location>/<timestamp>_board_setPoint_<V>V_fan_setPoint_<fan>_mfc_setPoint_<Gas>_<ppm>ppm_p<k>

One file = one recording, a text table with 92 columns and ~13,000-26,000 rows over 260 s:
    col 0        time in ms
    col 1-2      fan set point / fan speed
    col 3-5      gas flow set points: 0 before the release, > 0 while gas is released (about 20 s to 200 s)
    col 6-8      measured gas flows
    col 9-10     temperature, relative humidity
    col 11-91    9 boards x (1 flag column + 8 sensor columns)  ->  72 sensor columns
"""
import re
import zipfile

import numpy as np
import pandas as pd

GASES = ["Acetaldehyde", "Acetone", "Ammonia", "Benzene", "Butanol",
         "CO", "Ethylene", "Methane", "Methanol", "Toluene"]
# positive class of the binary task
CO_CLASS = GASES.index("CO")
FANS = [0, 60, 100]                           # the 3 wind speed set points
# 72 column indices
SENSOR_COLS = [11 + 9 * b + 1 + s for b in range(9) for s in range(8)]
# board 1-9, sensor 1-8 on the board
SENSOR_NAMES = [f"B{b + 1}_S{s + 1}" for b in range(9) for s in range(8)]
RELEASE_COL = 3                               # gas flow set point column

_PATTERN = re.compile(
    r"WTD_upload/(?P<folder>[^/]+)/L(?P<location>\d)/(?P<timestamp>\d+)_board_setPoint_(?P<voltage>\d+)V"
    r"_fan_setPoint_(?P<fan>\d+)_mfc_setPoint_(?P<gas>[A-Za-z]+)_(?P<ppm>\d+)ppm_p(?P<p>\d+)$")


def list_recordings(zip_path):
    """Parse all recording file names of the zip (no file is read).

    returns DataFrame, one row per recording:
        path, folder, gas, ppm, location, voltage, fan, timestamp
    """
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
    rows = []
    for name in names:
        m = _PATTERN.search(name)
        if m:
            rows.append({"path": name, "folder": m["folder"], "gas": m["gas"],
                         "ppm": int(m["ppm"]), "location": int(m["location"]),
                         "voltage": int(m["voltage"]), "fan": int(m["fan"]),
                         "timestamp": m["timestamp"]})
    return pd.DataFrame(rows)


def read_recording(zip_file, path):
    """Read one recording from an open zipfile.ZipFile.

    returns
        t        (T_raw,)     time in seconds
        x        (72, T_raw)  sensor values
        release  (start, end) gas release times in seconds (from the flow set point column)
    """
    with zip_file.open(path) as f:
        # (T_raw, 92)
        a = pd.read_csv(f, sep=r"\s+", header=None).to_numpy(dtype=float)
    # (T_raw,)
    t = a[:, 0] / 1000.0
    # (72, T_raw)
    x = a[:, SENSOR_COLS].T
    # rows while gas flows
    on = np.flatnonzero(a[:, RELEASE_COL] > 0)
    release = (t[on[0]], t[on[-1]]) if len(on) else (np.nan, np.nan)
    return t, x, release
