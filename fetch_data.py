"""Download real historical klines (1h) from Binance public data archive."""
import io
import os
import sys
import zipfile
import urllib.request

import pandas as pd

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
URL = "https://data.binance.vision/data/spot/monthly/klines/{sym}/1h/{sym}-1h-{ym}.zip"

COLS = ["time", "open", "high", "low", "close", "volume"]


def months(start="2023-01", end="2026-08"):
    y, m = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    out = []
    while (y, m) <= (ey, em):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def download_one(sym, ym):
    url = URL.format(sym=sym, ym=ym)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f, header=None)
    df = df.iloc[:, :6]
    df.columns = COLS
    # open time: ms epoch for spot
    ts = df["time"]
    unit = "ms" if ts.iloc[0] < 1e14 else "us"
    df["time"] = pd.to_datetime(ts, unit=unit)
    df["symbol"] = sym
    return df


def get_symbol(sym, start, end):
    os.makedirs(BASE, exist_ok=True)
    cache = os.path.join(BASE, f"{sym}_1h.csv.gz")
    if os.path.exists(cache):
        print(f"cache hit {cache}")
        df = pd.read_csv(cache, compression="gzip")
        df["time"] = pd.to_datetime(df["time"], format="mixed",
                                    errors="coerce")
        return df[df["time"].notna()]
    frames, missing = [], []
    for ym in months(start, end):
        try:
            frames.append(download_one(sym, ym))
            print(f"{sym} {ym} ok", flush=True)
        except Exception as e:
            missing.append((ym, str(e)))
            print(f"{sym} {ym} FAIL {e}", flush=True)
    if not frames:
        raise SystemExit(f"no data for {sym}")
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)
    df.to_csv(cache, index=False, compression="gzip")
    if missing:
        print(f"{sym}: {len(missing)} months missing: {[m[0] for m in missing]}")
    return df


if __name__ == "__main__":
    syms = sys.argv[1].split(",") if len(sys.argv) > 1 else [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"
    ]
    start = sys.argv[2] if len(sys.argv) > 2 else "2023-01"
    end = sys.argv[3] if len(sys.argv) > 3 else "2026-08"
    for s in syms:
        df = get_symbol(s, start, end)
        print(f"{s}: {len(df)} bars  {df['time'].iloc[0]} -> {df['time'].iloc[-1]}")
