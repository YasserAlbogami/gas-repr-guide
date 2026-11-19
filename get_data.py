"""Put both datasets into data/raw/. Run once after cloning:  python get_data.py

1. EverydayScents: unzips data/everyday_scents.zip (committed) into data/raw/everyday_scents/.
2. GasWindTunnel: downloads the UCI 251 zip into data/raw/gas_wind_tunnel/ and unzips it.
   The download resumes if it was interrupted; run the script again to continue.
"""
import os
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent
ES_ZIP = ROOT / "data" / "everyday_scents.zip"
ES_DIR = ROOT / "data" / "raw" / "everyday_scents"
GWT_DIR = ROOT / "data" / "raw" / "gas_wind_tunnel"
GWT_ZIP = GWT_DIR / "uci251.zip"
GWT_URL = ("https://archive.ics.uci.edu/static/public/251/"
           "gas+sensor+arrays+in+open+sampling+settings.zip")


def unzip_flat(zip_path, out_dir):
    """Extract every file of zip_path into out_dir, dropping the folders inside the zip."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        for member in z.infolist():
            name = os.path.basename(member.filename)
            if member.is_dir() or not name:
                continue
            with z.open(member) as src, open(out_dir / name, "wb") as dst:
                dst.write(src.read())


def download(url, path):
    """Download url to path, resuming from the bytes already on disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    done = path.stat().st_size if path.exists() else 0
    request = urllib.request.Request(url, headers={"Range": f"bytes={done}-"})
    with urllib.request.urlopen(request) as response:
        if response.status == 200:  # server ignored the range: start over
            done = 0
        mode = "ab" if done else "wb"
        with open(path, mode) as f:
            while chunk := response.read(1 << 20):
                f.write(chunk)
                done += len(chunk)
                print(f"\r  {done / 1e9:.2f} GB", end="", flush=True)
    print()


def main():
    print("EverydayScents ...")
    unzip_flat(ES_ZIP, ES_DIR)

    print("GasWindTunnel download ...")
    try:
        download(GWT_URL, GWT_ZIP)
    except urllib.error.HTTPError as e:
        if e.code != 416:  # 416 = file already complete
            raise
    print("GasWindTunnel unzip ...")
    with zipfile.ZipFile(GWT_ZIP) as z:
        z.extractall(GWT_DIR)
    print("done")


if __name__ == "__main__":
    main()
