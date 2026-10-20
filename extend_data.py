"""
Extend the 1h cache deeper into history and add new symbols.

- downloads ONLY missing months from data.binance.vision (monthly 1h zips)
- existing cache rows are never deleted or rewritten (merge + dedup on time)
- months that do not exist (pre-listing, delisted, current month) are
  skipped silently and counted

usage:
  python extend_data.py            # all old + new symbols, 2018-01 .. 2026-09
"""
import os
import sys

import pandas as pd

import backtest as bt
import fetch_data as fd

START = "2018-01"
END = "2026-09"

NEW_SYMS = [
    "AAVEUSDT", "GRTUSDT", "HBARUSDT", "ICPUSDT", "SANDUSDT",
    "AXSUSDT", "MANAUSDT", "EGLDUSDT", "NEOUSDT", "VETUSDT",
    "KAVAUSDT", "ENJUSDT", "ZECUSDT", "1INCHUSDT", "CHZUSDT",
]


def extend(sym, start=START, end=END):
    cache = os.path.join(fd.BASE, f"{sym}_1h.csv.gz")
    have = pd.DataFrame()
    if os.path.exists(cache):
        have = pd.read_csv(cache, compression="gzip")
        have["time"] = pd.to_datetime(have["time"], format="mixed",
                                      errors="coerce")
        have = have[have["time"].notna()]
    if len(have) and "symbol" not in have.columns:
        have["symbol"] = sym
    have_months = (set(have["time"].dt.strftime("%Y-%m"))
                   if len(have) else set())
    want = [ym for ym in fd.months(start, end) if ym not in have_months]

    frames, miss = [], 0
    for ym in want:
        try:
            frames.append(fd.download_one(sym, ym))
            print(f"  {sym} {ym} ok", flush=True)
        except Exception:
            miss += 1
    if frames:
        new = pd.concat(frames, ignore_index=True)
        df = (pd.concat([have, new], ignore_index=True, sort=False)
              if len(have) else new)
        df = df.drop_duplicates(subset="time").sort_values("time")
        df = df.reset_index(drop=True)
        df.to_csv(cache, index=False, compression="gzip")
    else:
        df = have
    return df, len(want), len(frames), miss


def main():
    syms = list(bt.SYMBOLS)      # extended list already includes NEW_SYMS
    only = sys.argv[1].split(",") if len(sys.argv) > 1 else None
    if only:
        syms = only
    print(f"symbols: {len(syms)}  window: {START} .. {END}", flush=True)
    summary = []
    for sym in syms:
        try:
            df, want, got, miss = extend(sym)
        except Exception as e:
            print(f"{sym}: FATAL {e}", flush=True)
            summary.append((sym, 0, "-", "-", "FATAL"))
            continue
        if not len(df):
            print(f"{sym}: NO DATA AT ALL (skipped)", flush=True)
            summary.append((sym, 0, "-", "-", "no data"))
            continue
        summary.append((sym, len(df), str(df["time"].iloc[0])[:10],
                        str(df["time"].iloc[-1])[:10],
                        f"new {got}/{want} (miss {miss})"))
        print(f"{sym}: {len(df)} bars  {summary[-1][2]} -> {summary[-1][3]}  "
              f"({summary[-1][4]})", flush=True)

    print("\n==== coverage ====")
    for row in summary:
        print(f"  {row[0]:10s} {row[1]:7d} bars  {row[2]} -> {row[3]}  {row[4]}")
    ok = sum(1 for r in summary if r[1] > 0)
    print(f"\n{ok}/{len(summary)} symbols have data")


if __name__ == "__main__":
    main()
