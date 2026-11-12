"""Download the UCI Diabetes 130-US hospitals data (CC BY 4.0) into diabetes+130-us+hospitals+for+years+1999-2008/ in the repo root.

The data already ships with the repo, so you only need this to re-download it.

    python data/fetch_data.py                 # download from UCI
    python data/fetch_data.py --zip path.zip  # use a zip you already downloaded
"""
import argparse
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = ("https://archive.ics.uci.edu/static/public/296/"
       "diabetes+130-us+hospitals+for+years+1999-2008.zip")
RAW = Path(__file__).resolve().parents[1] / "diabetes+130-us+hospitals+for+years+1999-2008"
EXPECTED_ROWS = 101_766
EXPECTED_COLS = 50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, help="local copy of the UCI zip")
    args = ap.parse_args()

    if args.zip:
        blob = args.zip.read_bytes()
    else:
        print(f"Downloading {URL}")
        with urllib.request.urlopen(URL, timeout=120) as r:
            blob = r.read()

    RAW.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for name in z.namelist():
            base = Path(name).name
            if base in {"diabetic_data.csv", "IDS_mapping.csv"}:
                (RAW / base).write_bytes(z.read(name))

    csv = RAW / "diabetic_data.csv"
    with csv.open() as f:
        header = f.readline().strip().split(",")
        rows = sum(1 for _ in f)
    ok = rows == EXPECTED_ROWS and len(header) == EXPECTED_COLS
    print(f"{csv}: {rows} rows, {len(header)} columns -> {'OK' if ok else 'UNEXPECTED'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
