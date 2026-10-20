"""
Stage 5: explicit Reward:Risk search.

Adds exit geometries built for RR ~ 2-3 (planned TP = k * SL, k = 2 or 3)
plus a low-fraction runner variant. Signals grid mirrors stage4.

Break-even win rate reminder: at RR=2 the break-even WR is 33.3%,
at RR=3 it is 25% -- so a profitable RR-2 config must beat 33% WR.
"""
import itertools
import os
import time

import numpy as np
import pandas as pd

import backtest as bt

RESULTS = bt.RESULTS

TSTOP_ONE = {"1h": 72, "4h": 30, "1d": 15}

# planned-RR pairs: (sl_atr, tp_atr)  -> realized RR ~= tp/sl
FIXED_PAIRS = [(1.0, 2.0), (1.5, 3.0), (2.0, 4.0),
               (1.0, 3.0), (1.5, 4.5)]

# low-fraction runner variants (mostly runner, small first take)
TRAIL_LITE = list(itertools.product([0.15, 0.3], [0.4], [2.0], [1.5],
                                    [8.0]))  # tp1, frac1, trail, sl, tp2


def signals_for(tf):
    bb_len = [14, 30] if tf == "1h" else [14, 20, 30]
    return list(itertools.product(["touch", "reclaim"], bb_len,
                                  [1.5, 2.5], [35, 50],
                                  ["none", "ema200"]))


def stage5(store):
    rows = []
    t0 = time.time()
    done = 0
    for tf in ("4h", "1d", "1h"):
        sigs = signals_for(tf)
        for (entry, bbl, bbd, rsi, filt) in sigs:
            for (sl, tp) in FIXED_PAIRS:
                p = dict(tf=tf, entry=entry, bb_len=bbl, bb_dev=bbd,
                         rsi_th=rsi, filter=filt, exit="fixed",
                         tp_atr=tp, sl_atr=sl, time_stop=TSTOP_ONE[tf])
                _, tr, te = bt.run_config(store, p)
                rows.append({**p, **{f"train_{k}": v for k, v in tr.items()},
                             **{f"test_{k}": v for k, v in te.items()}})
                done += 1
            if tf != "1h":  # trail-lite only on the fast timeframes
                for (frac1, tp1, trail, sl, tp2) in TRAIL_LITE:
                    p = dict(tf=tf, entry=entry, bb_len=bbl, bb_dev=bbd,
                             rsi_th=rsi, filter=filt, exit="trail",
                             tp1_atr=tp1, be_atr=0.8, trail_atr=trail,
                             tp2_atr=tp2, frac1=frac1, sl_atr=sl,
                             time_stop=TSTOP_ONE[tf])
                    _, tr, te = bt.run_config(store, p)
                    rows.append({**p,
                                 **{f"train_{k}": v for k, v in tr.items()},
                                 **{f"test_{k}": v for k, v in te.items()}})
                    done += 1
            if done % 60 == 0:
                print(f"  stage5 {done} configs  {time.time()-t0:.0f}s",
                      flush=True)
        print(f"  stage5 {tf} done ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(os.path.join(RESULTS, "stage5.csv"), index=False)

    cols = ["tf", "entry", "exit", "bb_len", "bb_dev", "rsi_th", "filter",
            "tp_atr", "sl_atr", "tp1_atr", "frac1", "trail_atr", "tp2_atr",
            "train_n", "train_win_rate", "train_rr", "train_pf",
            "train_expectancy", "test_n", "test_win_rate", "test_rr",
            "test_pf", "test_expectancy"]
    print("\nBest WR with RR >= 1.8, PF >= 1.05, n >= 300 (train):")
    sub = df[(df.train_rr >= 1.8) & (df.train_pf >= 1.05)
             & (df.train_n >= 300)]
    if sub.empty:
        print("  NONE")
    else:
        b = sub.sort_values("train_win_rate", ascending=False)
        print(b[cols].head(10).to_string(index=False))
        best = b.iloc[0]
        print(f"\n  -> train WR={best.train_win_rate:.1f}% "
              f"RR={best.train_rr:.2f} PF={best.train_pf:.2f} | "
              f"test WR={best.test_win_rate:.1f}% RR={best.test_rr:.2f} "
              f"PF={best.test_pf:.2f} n={int(best.test_n)}")
    return df


if __name__ == "__main__":
    print("loading ...")
    store = {tf: {s: bt.load_symbol(s, tf) for s in bt.SYMBOLS}
             for tf in ("1h", "4h", "1d")}
    stage5(store)
