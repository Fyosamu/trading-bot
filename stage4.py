"""
Stage 4: wide search targeting Reward:Risk (avg_win / |avg_loss|) ~ 2
with the best possible win rate, >= 1000 trades overall.

Exit model 'trail' (unit-tested in test_sim.py):
  close frac1 at tp1*ATR -> stop to breakeven -> ratchet stop behind the
  running best price by trail_atr*ATR (conservative: judged with the best
  price reached BEFORE the current bar), optional hard target tp2*ATR
  (0 = none), hard SL sl_atr*ATR, time stop.

Selection uses TRAIN data only.
"""
import itertools
import os
import time

import numpy as np
import pandas as pd

import backtest as bt

RESULTS = bt.RESULTS

TSTOP_ONE = {"1h": 72, "4h": 30, "1d": 15}

# (tp1, frac1, trail, sl, tp2)
EXITS_FULL = list(itertools.product([0.4, 1.0, 1.6], [0.5, 0.7],
                                    [1.5, 3.0], [1.5, 2.5],
                                    [0.0, 8.0]))
# lighter set for the slow 1h timeframe
EXITS_LITE = list(itertools.product([0.4, 1.6], [0.7],
                                    [1.5, 3.0], [2.0],
                                    [0.0, 8.0]))


def signals_for(tf):
    bb_len = [14, 30] if tf == "1h" else [14, 20, 30]
    return list(itertools.product(["touch", "reclaim"], bb_len,
                                  [1.5, 2.5], [35, 50],
                                  ["none", "ema200"]))


def stage4(store):
    rows = []
    t0 = time.time()
    done = 0
    for tf in ("4h", "1d", "1h"):
        exits = EXITS_LITE if tf == "1h" else EXITS_FULL
        for (entry, bbl, bbd, rsi, filt) in signals_for(tf):
            for (tp1, frac1, trail, sl, tp2) in exits:
                p = dict(tf=tf, entry=entry, bb_len=bbl, bb_dev=bbd,
                         rsi_th=rsi, filter=filt, exit="trail",
                         tp1_atr=tp1, be_atr=0.8, trail_atr=trail,
                         tp2_atr=tp2, frac1=frac1, sl_atr=sl,
                         time_stop=TSTOP_ONE[tf])
                _, tr, te = bt.run_config(store, p)
                rows.append({**p, **{f"train_{k}": v for k, v in tr.items()},
                             **{f"test_{k}": v for k, v in te.items()}})
                done += 1
                if done % 60 == 0:
                    print(f"  stage4 {done} configs  {time.time()-t0:.0f}s",
                          flush=True)
        print(f"  stage4 {tf} done ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(os.path.join(RESULTS, "stage4.csv"), index=False)

    cols = ["tf", "entry", "bb_len", "bb_dev", "rsi_th", "filter",
            "tp1_atr", "be_atr", "trail_atr", "tp2_atr", "frac1", "sl_atr",
            "train_n", "train_win_rate", "train_rr", "train_pf",
            "train_expectancy", "test_n", "test_win_rate", "test_rr",
            "test_pf", "test_expectancy"]

    # ---- Pareto view (train only): best WR subject to an RR floor -------
    print("\nPareto (train): best win rate subject to RR floor")
    for floor in (1.0, 1.5, 2.0, 2.5, 3.0):
        sub = df[(df.train_rr >= floor) & (df.train_n >= 550)
                 & (df.train_pf >= 1.05)]
        if sub.empty:
            print(f"  RR>={floor}: none")
            continue
        best = sub.loc[sub.train_win_rate.idxmax()]
        print(f"  RR>={floor}: WR={best.train_win_rate:.1f}% "
              f"RR={best.train_rr:.2f} PF={best.train_pf:.2f} "
              f"n={int(best.train_n)} exp={best.train_expectancy:.3f}% "
              f"| test WR={best.test_win_rate:.1f}% RR={best.test_rr:.2f} "
              f"PF={best.test_pf:.2f} n={int(best.test_n)} "
              f"| {best.tf}/{best['entry']} bb{int(best.bb_len)}/"
              f"{best.bb_dev} rsi{int(best.rsi_th)} {best['filter']} "
              f"tp1={best.tp1_atr} f={best.frac1} tr={best.trail_atr} "
              f"tp2={best.tp2_atr} sl={best.sl_atr}")

    # ---- shortlist for the report ---------------------------------------
    tiers = ((550, 1.05, 1.8), (400, 1.0, 1.5), (300, 1.0, 1.2))
    d = pd.DataFrame()
    for (n, pf, rr) in tiers:
        d = df[(df.train_n >= n) & (df.train_pf >= pf)
               & (df.train_rr >= rr)]
        if not d.empty:
            print(f"\nshortlist tier: n>={n} pf>={pf} rr>={rr}")
            break
    if d.empty:
        d = df
    d = d.assign(score=d.train_win_rate + 5.0 * np.minimum(d.train_pf, 3.0))
    d = d.sort_values("score", ascending=False)
    print(f"STAGE4 shortlist: {len(d)} configs")
    print(d[cols].head(20).to_string(index=False))
    d[cols].to_csv(os.path.join(RESULTS, "stage4_top.csv"), index=False)
    return df


if __name__ == "__main__":
    print("loading ...")
    store = {tf: {s: bt.load_symbol(s, tf) for s in bt.SYMBOLS}
             for tf in ("1h", "4h", "1d")}
    stage4(store)
