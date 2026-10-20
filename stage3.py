"""
Stage 3: focused search around the winning family
(reclaim entry + scale-out exit) on ALL timeframes, aiming for
train_n high enough that the full backtest clears 1000 trades
while keeping win rate >= 85% and PF > 1.

Selection still uses TRAIN only.
"""
import itertools
import os
import time

import numpy as np
import pandas as pd

import backtest as bt

RESULTS = bt.RESULTS


def stage3(store):
    rows = []
    t0 = time.time()
    done = 0
    for tf in ("1h", "4h", "1d"):
        for (entry, bbl, bbd, rsi, filt, ts, tp1, tp2, sl) in itertools.product(
                ["touch", "reclaim"], [14, 20, 30], [1.5, 2.0, 2.5],
                [35, 50], ["none", "ema200"], bt.TSTOP[tf],
                [0.3, 0.5], [1.5, 2.5], [3.0, 4.0]):
            p = dict(tf=tf, entry=entry, bb_len=bbl, bb_dev=bbd,
                     rsi_th=rsi, filter=filt, exit="scale",
                     tp1_atr=tp1, be_atr=0.8, tp2_atr=tp2, frac1=0.6,
                     sl_atr=sl, time_stop=ts)
            _, tr, te = bt.run_config(store, p)
            rows.append({**p, **{f"train_{k}": v for k, v in tr.items()},
                         **{f"test_{k}": v for k, v in te.items()}})
            done += 1
            if done % 60 == 0:
                print(f"  stage3 {done} configs  {time.time()-t0:.0f}s",
                      flush=True)
        print(f"  stage3 {tf} done ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(os.path.join(RESULTS, "stage3.csv"), index=False)

    # ---- shortlist: enough trades, WR >= 85 on train, PF >= 1.05 ----------
    d = df[(df.train_n >= 550) & (df.train_win_rate >= 85.0)
           & (df.train_pf >= 1.05)].copy()
    if d.empty:
        d = df[(df.train_n >= 400) & (df.train_win_rate >= 83.0)
               & (df.train_pf >= 1.0)].copy()
    d["score"] = d.train_win_rate + 5.0 * np.minimum(d.train_pf, 3.0)
    d = d.sort_values("score", ascending=False)
    cols = ["tf", "entry", "bb_len", "bb_dev", "rsi_th", "filter",
            "time_stop",
            "tp1_atr", "be_atr", "tp2_atr", "sl_atr",
            "train_n", "train_win_rate", "train_pf", "train_expectancy",
            "test_n", "test_win_rate", "test_pf", "test_expectancy"]
    print(f"\nSTAGE3 shortlist: {len(d)} configs meet the bar")
    print(d[cols].head(20).to_string(index=False))
    d[cols].to_csv(os.path.join(RESULTS, "stage3_top.csv"), index=False)
    return df


if __name__ == "__main__":
    print("loading ...")
    store = {tf: {s: bt.load_symbol(s, tf) for s in bt.SYMBOLS}
             for tf in ("1h", "4h", "1d")}
    for tf in store:
        print(f"  {tf}: {len(bt.SYMBOLS)} symbols x "
              f"{len(store[tf][bt.SYMBOLS[0]]['close'])} bars")
    stage3(store)
