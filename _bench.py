"""Bench: how long does one run_config take on the extended store?
Also validates build_store + clean + phase wiring before the full run."""
import time

import backtest as bt
import find_best as fb

t0 = time.time()
store = fb.build_store()
print(f"store build: {time.time()-t0:.0f}s")

cfgs = []
for es in fb.entry_sets()[:3]:
    for tf in fb.TFS:
        p = dict(es)
        p.update(fb.exit_presets()[3])
        p.update(tf=tf, filter="ema200", time_stop=bt.TSTOP[tf][1])
        cfgs.append(fb.clean(p))

t0 = time.time()
for p in cfgs:
    _, tr, te = bt.run_config(store, p)
dt = (time.time() - t0) / len(cfgs)
n1 = 27 * 2 * 3 * 6 + 1
print(f"per config: {dt:.2f}s  ->  N1 ({n1} configs) ~ {dt*n1/60:.0f} min")
print(f"sample train n={tr['n']} wr={tr['win_rate']:.1f} pf={tr['pf']:.2f}")
