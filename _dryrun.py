"""Dry-run of the full selection pipeline on a tiny config slice.
Validates: phase -> csv -> rank -> picks -> param_dict round-trip ->
reeval -> subperiods -> json. Deletes its own output afterwards."""
import json
import os

import backtest as bt
import find_best as fb

store = fb.build_store()

cfgs = [fb.v1_baseline()]
for es in fb.entry_sets()[:4]:
    p = dict(es)
    p.update(fb.exit_presets()[3])
    p.update(tf="4h", filter="ema200", time_stop=bt.TSTOP["4h"][1])
    cfgs.append(p)

df = fb.phase(store, cfgs, "dry")
d = fb.train_rank(df)
cols = ["entry", "tf", "train_n", "train_win_rate", "train_pf", "train_rr"]
print(d[cols].head(10).to_string(index=False))

pks, n_elig = fb.picks_on_train(df)
print(f"eligible: {n_elig}")
for k in ("A", "B", "C"):
    r = pks.get(k)
    print(f"  {k}: " + (f"{r['entry']} wr={r['train_win_rate']:.1f} "
                        f"rr={r['train_rr']:.2f}" if r else "none"))

row = pks.get("B") or pks.get("C") or pks.get("A") or d.iloc[0].to_dict()
p = fb.param_dict(row)
print("param_dict:", json.dumps(p))
trades, tr, te = fb.reeval(store, p)
sp = fb.subperiods(trades)
print(f"reeval: train n={tr['n']} wr={tr['win_rate']:.1f} | "
      f"test n={te['n']} wr={te['win_rate']:.1f}")
print("subperiods:", sp)
json.dumps({"params": p, "train": tr, "test": te}, default=str)
print("json OK")

path = os.path.join(fb.RESULTS, "v2_dry.csv")
if os.path.exists(path):
    os.remove(path)
    print("cleaned v2_dry.csv")
print("DRY RUN PASSED")
