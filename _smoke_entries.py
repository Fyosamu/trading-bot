"""Smoke test: every entry mode builds signals and simulates cleanly.
Retries on load because extend_data.py may be rewriting caches right now."""
import time

import backtest as bt

sd = None
for sym in ("ETHUSDT", "SOLUSDT", "XRPUSDT", "BTCUSDT", "ADAUSDT"):
    for _ in range(4):
        try:
            sd = bt.load_symbol(sym, "4h")
            break
        except Exception:
            time.sleep(3)
    if sd is not None:
        print(f"loaded {sym}")
        break
if sd is None:
    raise SystemExit("could not load any symbol (all caches busy?)")
cases = [
    dict(tf="4h", entry="touch", bb_len=30, bb_dev=2.5, rsi_th=50,
         filter="ema200", exit="fixed", tp_atr=2, sl_atr=2, time_stop=30),
    dict(tf="4h", entry="reclaim", bb_len=20, bb_dev=2.0, rsi_th=50,
         filter="none", exit="fixed", tp_atr=2, sl_atr=2, time_stop=30),
    dict(tf="4h", entry="rsi_rev", bb_len=30, bb_dev=2.5, rsi_th=30,
         filter="ema200", exit="fixed", tp_atr=2, sl_atr=2, time_stop=30),
    dict(tf="4h", entry="donchian", bb_len=30, bb_dev=2.5, rsi_th=50,
         filter="ema200", exit="fixed", dc_len=40, tp_atr=3, sl_atr=2,
         time_stop=30),
    dict(tf="4h", entry="ema_pull", bb_len=30, bb_dev=2.5, rsi_th=50,
         filter="ema200", exit="fixed", ema_f=20, tp_atr=3, sl_atr=2,
         time_stop=30),
    dict(tf="4h", entry="momentum", bb_len=30, bb_dev=2.5, rsi_th=50,
         filter="ema200", exit="fixed", mom_k=24, mom_x=1.0, tp_atr=3,
         sl_atr=2, time_stop=30),
    dict(tf="4h", entry="donchian", bb_len=30, bb_dev=2.5, rsi_th=50,
         filter="none", exit="trail", dc_len=20, tp1_atr=0.3, sl_atr=4.0,
         frac1=0.1, tp2_atr=0, trail_atr=1.0, time_stop=30),
]
for p in cases:
    d = bt.build_dirs(sd, p)
    m = bt.metrics(bt.simulate(d, sd, p))
    print(f"{p['entry']:9s} dirs: L={int((d == 1).sum()):4d} "
          f"S={int((d == -1).sum()):4d}  trades={m['n']:4d} "
          f"wr={m['win_rate']:5.1f} pf={m['pf']:.2f}")
print("smoke OK")
