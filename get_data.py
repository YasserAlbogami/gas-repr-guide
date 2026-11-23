"""Put both datasets into data/raw/. Run once after cloning:  python get_data.py

1. EverydayScents: unzips data/everyday_scents.zip (committed) into data/raw/everyday_scents/.
2. GasWindTunnel: downloads the UCI 251 zip (about 8 GB) to data/raw/gas_wind_tunnel/uci251.zip.
   The zip is NOT unzipped (that would take about 100 GB): notebook 00a reads only the
   600 recordings it needs straight from the zip.
   The UCI server does not support resuming, so an interrupted download starts over.
"""
import os
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent
ES_ZIP = ROOT / "data" / "everyday_scents.zip"
ES_DIR = ROOT / "data" / "raw" / "everyday_scents"
GWT_ZIP = ROOT / "data" / "raw" / "gas_wind_tunnel" / "uci251.zip"
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


def is_complete_zip(path):
    """True if path is a readable zip (a partial download has no central directory)."""
    try:
        with zipfile.ZipFile(path):
            return True
    except (zipfile.BadZipFile, FileNotFoundError):
        return False


def download(url, path):
    """Download url to path through a temporary .part file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(".part")
    done = 0
    with urllib.request.urlopen(url) as response, open(part, "wb") as f:
        while chunk := response.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            print(f"\r  {done / 1e9:.2f} GB", end="", flush=True)
    print()
    part.replace(path)


def main():
    print("EverydayScents: unzip ...")
    unzip_flat(ES_ZIP, ES_DIR)

    if is_complete_zip(GWT_ZIP):
        print("GasWindTunnel: zip already complete, skipping download")
    else:
        print("GasWindTunnel: download (about 8 GB) ...")
        download(GWT_URL, GWT_ZIP)
        assert is_complete_zip(GWT_ZIP), "download finished but the zip is not readable"
    print("done")


if __name__ == "__main__":
    main()
