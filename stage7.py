"""
Stage 7: break the WR ceiling at Reward:Risk >= 1.5.

Current ceiling from stages 4-6: train WR ~47% at rr>=1.5 (n>=400).
New levers here (never searched before):
  * 'scale' exit   -- partial at tp1, breakeven, HARD runner target tp2
                      (no trailing), big avg_win from tp2 hits
  * time_stop x2   -- 4h: 60 bars, 1d: 30 bars (runners need room)
  * sl 2.0 / 2.5 / 4.0 spread (smaller SL helps avg_loss, wider helps WR)
Full 4h + 1d signal grid (96 per timeframe).

Objective (TRAIN only): maximize win rate subject to
train_rr >= 1.5, train_pf >= 1.05, train_n >= 400.
"""
import itertools
import os
import time

import pandas as pd

import backtest as bt

RESULTS = bt.RESULTS

TSTOP_BASE = {"4h": 30, "1d": 15}

SIGNALS = list(itertools.product(["touch", "reclaim"], [14, 20, 30],
                                  [1.5, 2.5], [35, 50],
                                  ["none", "ema200"]))          # 96 / tf

# (exit, kwargs, tstop_mult)
EXITS = []
# trail: small first take + tight trail, various SL, optional hard target
for trail, sl, tp2, mult in itertools.product([1.0, 2.0], [2.0, 2.5, 4.0],
                                              [0.0, 8.0], [1, 2]):
    EXITS.append(("trail", dict(tp1_atr=0.3, be_atr=0.4, frac1=0.1,
                                trail_atr=trail, sl_atr=sl, tp2_atr=tp2),
                  mult))
# scale: hard runner target instead of trailing
for tp2, sl, mult in itertools.product([4.0, 6.0], [2.0, 3.0], [1, 2]):
    EXITS.append(("scale", dict(tp1_atr=0.3, be_atr=0.4, frac1=0.1,
                                tp2_atr=tp2, sl_atr=sl), mult))
# fixed TP = 2x / 3x SL with long time stop
for tp, sl, mult in itertools.product([2.0, 3.0], [1.0, 1.5], [1, 2]):
    EXITS.append(("fixed", dict(tp_atr=tp, sl_atr=sl), mult))
# tighter first target, mid trail
for mult in (1, 2):
    EXITS.append(("trail", dict(tp1_atr=0.15, be_atr=0.4, frac1=0.1,
                                trail_atr=1.5, sl_atr=3.0, tp2_atr=0.0),
                  mult))


def stage7(store):
    rows = []
    t0 = time.time()
    done = 0
    for tf in ("4h", "1d"):
        for (entry, bbl, bbd, rsi, filt) in SIGNALS:
            for (exit_mode, kw, mult) in EXITS:
                p = dict(tf=tf, entry=entry, bb_len=bbl, bb_dev=bbd,
                         rsi_th=rsi, filter=filt, exit=exit_mode,
                         time_stop=TSTOP_BASE[tf] * mult, **kw)
                _, tr, te = bt.run_config(store, p)
                rows.append({**p, **{f"train_{k}": v for k, v in tr.items()},
                             **{f"test_{k}": v for k, v in te.items()}})
                done += 1
            if done % 960 == 0:
                print(f"  stage7 {done} configs  {time.time()-t0:.0f}s",
                      flush=True)
        print(f"  stage7 {tf} done ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(os.path.join(RESULTS, "stage7.csv"), index=False)

    cols = ["tf", "entry", "exit", "bb_len", "bb_dev", "rsi_th", "filter",
            "tp1_atr", "be_atr", "frac1", "trail_atr", "tp2_atr", "tp_atr",
            "sl_atr", "time_stop", "train_n", "train_win_rate", "train_rr",
            "train_pf", "test_win_rate", "test_rr", "test_pf", "test_n"]
    sub = df[(df.train_rr >= 1.5) & (df.train_pf >= 1.05)
             & (df.train_n >= 400)]
    print(f"\nrr>=1.5 & pf>=1.05 & n>=400: {len(sub)} configs "
          f"({time.time()-t0:.0f}s)")
    if not sub.empty:
        b = sub.sort_values("train_win_rate", ascending=False)
        print(b[cols].head(12).to_string(index=False))
        best = b.iloc[0]
        print(f"\n  -> train WR={best.train_win_rate:.1f}% "
              f"RR={best.train_rr:.2f} PF={best.train_pf:.2f} "
              f"n={int(best.train_n)} | test WR={best.test_win_rate:.1f}% "
              f"RR={best.test_rr:.2f} PF={best.test_pf:.2f} "
              f"n={int(best.test_n)}")
    return df


if __name__ == "__main__":
    print("loading ...", flush=True)
    store = {tf: {s: bt.load_symbol(s, tf) for s in bt.SYMBOLS}
             for tf in ("4h", "1d")}
    stage7(store)
