"""
Stage 6: targeted search for the best R:R WHILE keeping train WR >= 80%.

Focus: the dominant signal families of the WR>=80 zone, with new levers
never searched before -- earlier breakeven (be_atr), wider SL, finer
fractions/trails.  Selection: train only; report test transparently.
"""
import itertools
import os
import time

import pandas as pd

import backtest as bt

RESULTS = bt.RESULTS

TSTOP_ONE = {"4h": 30, "1d": 15}

# dominant WR>=80 signal families from stage4/stage5
SIGNALS = [
    # (tf, entry, bb_len, bb_dev, rsi_th, filter)
    ("1d", "reclaim", 14, 1.5, 50, "ema200"),
    ("1d", "reclaim", 20, 1.5, 50, "ema200"),
    ("1d", "reclaim", 30, 1.5, 50, "ema200"),
    ("1d", "reclaim", 30, 1.5, 35, "none"),
    ("1d", "reclaim", 20, 2.5, 50, "none"),
    ("4h", "touch", 30, 2.5, 50, "ema200"),
    ("4h", "touch", 30, 2.5, 35, "ema200"),
]

EXITS = list(itertools.product(
    [0.3, 0.45, 0.6],      # tp1_atr  (easy first take)
    [0.4, 0.6, 0.8],       # be_atr   (NEW: earlier breakeven)
    [0.1, 0.2, 0.4],       # frac1    (smaller first slice -> bigger runner)
    [1.0, 2.0, 3.5],       # trail_atr
    [1.5, 2.5, 4.0],       # sl_atr   (NEW: wider stops)
    [0.0, 8.0],            # tp2_atr
))


def stage6(store):
    rows = []
    t0 = time.time()
    done = 0
    for (tf, entry, bbl, bbd, rsi, filt) in SIGNALS:
        for (tp1, be, frac, trail, sl, tp2) in EXITS:
            p = dict(tf=tf, entry=entry, bb_len=bbl, bb_dev=bbd, rsi_th=rsi,
                     filter=filt, exit="trail", tp1_atr=tp1, be_atr=be,
                     trail_atr=trail, tp2_atr=tp2, frac1=frac, sl_atr=sl,
                     time_stop=TSTOP_ONE[tf])
            _, tr, te = bt.run_config(store, p)
            rows.append({**p, **{f"train_{k}": v for k, v in tr.items()},
                         **{f"test_{k}": v for k, v in te.items()}})
            done += 1
        if done % 486 == 0:
            print(f"  stage6 {done} configs  {time.time()-t0:.0f}s",
                  flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(os.path.join(RESULTS, "stage6.csv"), index=False)

    cols = ["tf", "entry", "bb_len", "bb_dev", "rsi_th", "filter",
            "tp1_atr", "be_atr", "frac1", "trail_atr", "tp2_atr", "sl_atr",
            "train_n", "train_win_rate", "train_rr", "train_pf",
            "test_win_rate", "test_rr", "test_pf", "test_n"]
    sub = df[(df.train_win_rate >= 80) & (df.train_pf >= 1.05)
             & (df.train_n >= 300)]
    print(f"\nWR>=80 & pf>=1.05 & n>=300: {len(sub)} configs "
          f"({time.time()-t0:.0f}s)")
    if not sub.empty:
        b = sub.sort_values("train_rr", ascending=False)
        print(b[cols].head(12).to_string(index=False))
        best = b.iloc[0]
        print(f"\n  -> train WR={best.train_win_rate:.1f}% "
              f"RR={best.train_rr:.3f} PF={best.train_pf:.2f} "
              f"n={int(best.train_n)} | test WR={best.test_win_rate:.1f}% "
              f"RR={best.test_rr:.2f} PF={best.test_pf:.2f} "
              f"n={int(best.test_n)}")
    return df


if __name__ == "__main__":
    print("loading ...", flush=True)
    store = {tf: {s: bt.load_symbol(s, tf) for s in bt.SYMBOLS}
             for tf in ("4h", "1d")}
    stage6(store)
