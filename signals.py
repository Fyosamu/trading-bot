"""
Live signal check: reads the SAME public Binance data archive used by the
backtest, rebuilds the final strategy on the latest available bars and prints
whether the last completed candle fires a signal.

usage:  python signals.py            (uses results/final_config.json)
"""
import io
import json
import os
import zipfile
import datetime as dt
import urllib.request

import numpy as np
import pandas as pd

import backtest as bt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

MONTHLY = "https://data.binance.vision/data/spot/monthly/klines/{s}/1h/{s}-1h-{ym}.zip"
DAILY = "https://data.binance.vision/data/spot/daily/klines/{s}/1h/{s}-1h-{d}.zip"


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f, header=None)
    df = df.iloc[:, :6]
    df.columns = ["time", "open", "high", "low", "close", "volume"]
    unit = "ms" if df["time"].iloc[0] < 1e14 else "us"
    df["time"] = pd.to_datetime(df["time"], unit=unit)
    return df


def recent_bars(sym, days=320):
    today = dt.date.today()
    start = today - dt.timedelta(days=days)
    frames = []
    # full months (before current month)
    ym0, ym1 = (start.year, start.month), (today.year, today.month)
    y, m = ym0
    while (y, m) < ym1:
        try:
            frames.append(_get(MONTHLY.format(s=sym, ym=f"{y:04d}-{m:02d}")))
        except Exception:
            pass
        m += 1
        if m == 13:
            y, m = y + 1, 1
    # current partial month, day by day (skip today: not published yet)
    d = dt.date(today.year, today.month, 1)
    while d < today:
        try:
            frames.append(_get(DAILY.format(s=sym, d=d.isoformat())))
        except Exception:
            pass
        d += dt.timedelta(days=1)
    if not frames:
        raise SystemExit("no recent data retrieved")
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset="time").sort_values("time")
    return df


def main():
    with open(os.path.join(RESULTS, "final_config.json")) as f:
        cfg = json.load(f)
    p = cfg["params"]
    print("strategy:", json.dumps(p, indent=2))

    print("fetching recent 1h bars from Binance public archive ...")
    df = recent_bars("BTCUSDT", days=350)
    # aggregate to the strategy timeframe
    df["bucket"] = df["time"].dt.floor(bt.TF_RULE[p["tf"]] or "1h")
    g = df.groupby("bucket", sort=True)
    out = pd.DataFrame({
        "time": g["time"].first(), "open": g["open"].first(),
        "high": g["high"].max(), "low": g["low"].min(),
        "close": g["close"].last(), "volume": g["volume"].sum(),
    }).reset_index(drop=True)
    for c in ("open", "high", "low", "close", "volume"):
        out[c] = out[c].astype(float)
    sd = bt._make_sd(out["time"].to_numpy(), out["open"].to_numpy(),
                     out["high"].to_numpy(), out["low"].to_numpy(),
                     out["close"].to_numpy(), out["volume"].to_numpy())
    sd["symbol"] = "BTCUSDT"

    if len(sd["close"]) < 250:
        raise SystemExit(f"only {len(sd['close'])} bars - not enough")

    dirs = bt.build_dirs(sd, p)
    i = len(dirs) - 1
    last = out.iloc[i]
    a = sd["atr"][i]
    c = float(last["close"])
    side = int(dirs[i])
    print(f"\nlast completed {p['tf']} bar: {last['time']}  close={c}  "
          f"ATR={a:.4f}")
    print(f"data lag: archive updates after the bar/day closes (UTC)")
    if side == 0:
        print("-> NO SIGNAL on the last closed bar")
        return
    if p["exit"] == "fixed":
        tp = c + side * p["tp_atr"] * a
        sl = c - side * p["sl_atr"] * a
        print(f"-> SIGNAL {'LONG' if side > 0 else 'SHORT'}  "
              f"TP≈{tp:.4f}  SL≈{sl:.4f}")
    else:
        tp1 = c + side * p["tp1_atr"] * a
        tp2 = c + side * p["tp2_atr"] * a
        sl = c - side * p["sl_atr"] * a
        print(f"-> SIGNAL {'LONG' if side > 0 else 'SHORT'}  "
              f"TP1≈{tp1:.4f} (close {p['frac1']*100:.0f}%)  "
              f"BE≈{c:.4f}  TP2≈{tp2:.4f}  SL≈{sl:.4f}")


if __name__ == "__main__":
    main()
