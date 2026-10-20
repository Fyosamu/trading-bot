"""
Two-stage strategy search on the EXTENDED data (2018-01 -> 2026-09,
42 symbols: 27 original + 15 added by extend_data.py).

Protocol fixed in advance (same honesty rules as v1):
  * every ranking / pick uses TRAIN only (entry_time < 2025-01-01)
  * the OOS test period is reported once, for the final picks only
  * score = train_win_rate + 5 * min(train_pf, 3)   (v1's rank rule)
  * filters for any pick: train_n >= 300, train_pf >= 1.3, finite rr
  * the v1 winner config is force-included and re-scored on the same data
  * disclosure: the 2025+ window was seen in v1 iterations -> treat the new
    test numbers as "first look for these exact configs", not virgin data

Stages:
  n1   broad scan: 6 entry families x 3 TF x 2 filters x 6 exit presets
  n2a  fine exits   around top N1 parents (train score)
  n2b  entry-param neighbours of the best N2a rows
  n2c  time-stop dimension
  final Pareto picks on train + Persian report + JSON configs

usage:  python find_best.py n1 | n2 | final | all
"""
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd

import backtest as bt
from extend_data import NEW_SYMS

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

SYMS = list(bt.SYMBOLS)          # extended to 42 symbols in backtest.py
TFS = ("1h", "4h", "1d")
MIN_TRAIN_N = 300
MIN_TRAIN_PF = 1.3

INT_KEYS = {"bb_len", "rsi_th", "time_stop", "dc_len", "ema_f", "mom_k"}
FLOAT_KEYS = {"bb_dev", "tp_atr", "sl_atr", "tp1_atr", "tp2_atr", "frac1",
              "trail_atr", "mom_x"}
STR_KEYS = {"tf", "entry", "filter", "exit"}


def clean(d):
    """Drop NaN/None entries and cast to proper types (CSV round-trip safe)."""
    out = {}
    for k, v in d.items():
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        if k in INT_KEYS:
            out[k] = int(v)
        elif k in FLOAT_KEYS:
            out[k] = float(v)
        elif k in STR_KEYS:
            out[k] = str(v)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------- store
def load_all_tf(sym):
    """Read the 1h cache once, aggregate to every TF (mirrors load_symbol)."""
    path = os.path.join(bt.DATA, f"{sym}_1h.csv.gz")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, compression="gzip")
    df["time"] = pd.to_datetime(df["time"], format="mixed", errors="coerce")
    if df["time"].isna().any():
        df = df[df["time"].notna()]
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    df = df.sort_values("time").reset_index(drop=True)
    out = {}
    for tf in TFS:
        rule = bt.TF_RULE[tf]
        if rule is None:
            g = df
        else:
            b = df.copy()
            b["bucket"] = b["time"].dt.floor(rule)
            grp = b.groupby("bucket", sort=True)
            g = pd.DataFrame({
                "time": grp["time"].first(), "open": grp["open"].first(),
                "high": grp["high"].max(), "low": grp["low"].min(),
                "close": grp["close"].last(), "volume": grp["volume"].sum(),
            }).reset_index(drop=True)
        out[tf] = bt._make_sd(g["time"].to_numpy(), g["open"].to_numpy(),
                              g["high"].to_numpy(), g["low"].to_numpy(),
                              g["close"].to_numpy(), g["volume"].to_numpy())
    return out


def build_store():
    store = {tf: {} for tf in TFS}
    missing = []
    t0 = time.time()
    for s in SYMS:
        try:
            per = load_all_tf(s)
        except FileNotFoundError:
            missing.append(s)
            continue
        except Exception:
            time.sleep(3)          # extend_data may be rewriting this cache
            try:
                per = load_all_tf(s)
            except Exception:
                missing.append(s)
                continue
        for tf in TFS:
            store[tf][s] = per[tf]
    print(f"store: " + ", ".join(f"{tf}={len(store[tf])}" for tf in TFS)
          + f"  missing={missing}  ({time.time()-t0:.0f}s)", flush=True)
    return store


# ------------------------------------------------------------ param space
def entry_sets():
    s = []
    for bbl in (20, 30):
        for bbd in (2.0, 2.5):
            for rs in (35, 50):
                s.append(dict(entry="touch", bb_len=bbl, bb_dev=bbd,
                              rsi_th=rs))
    for bbl in (20, 30):
        for bbd in (2.0, 2.5):
            s.append(dict(entry="reclaim", bb_len=bbl, bb_dev=bbd,
                          rsi_th=50))
    for rs in (25, 30, 35):
        s.append(dict(entry="rsi_rev", bb_len=30, bb_dev=2.5, rsi_th=rs))
    for k in (20, 40, 55):
        s.append(dict(entry="donchian", bb_len=30, bb_dev=2.5, rsi_th=50,
                      dc_len=k))
    for f in (10, 20, 50):
        s.append(dict(entry="ema_pull", bb_len=30, bb_dev=2.5, rsi_th=50,
                      ema_f=f))
    for k in (12, 24, 48):
        for x in (1.0, 1.5):
            s.append(dict(entry="momentum", bb_len=30, bb_dev=2.5, rsi_th=50,
                          mom_k=k, mom_x=x))
    return s


def exit_presets():
    return [
        dict(exit="fixed", tp_atr=2.0, sl_atr=2.0),
        dict(exit="fixed", tp_atr=3.0, sl_atr=2.0),
        dict(exit="fixed", tp_atr=4.0, sl_atr=4.0),
        dict(exit="trail", tp1_atr=0.3, sl_atr=4.0, frac1=0.1,
             tp2_atr=0.0, trail_atr=1.0),
        dict(exit="trail", tp1_atr=0.5, sl_atr=3.0, frac1=0.3,
             tp2_atr=0.0, trail_atr=1.5),
        dict(exit="scale", tp1_atr=1.0, tp2_atr=3.0, sl_atr=2.0,
             frac1=0.5),
    ]


ENTRY_DIMS = {
    "touch": {"bb_len": [14, 20, 30, 40], "bb_dev": [1.5, 2.0, 2.5, 3.0],
              "rsi_th": [25, 35, 50]},
    "reclaim": {"bb_len": [14, 20, 30, 40], "bb_dev": [1.5, 2.0, 2.5, 3.0],
                "rsi_th": [35, 50, 65]},
    "rsi_rev": {"rsi_th": [20, 25, 30, 35, 40]},
    "donchian": {"dc_len": [10, 20, 30, 40, 55, 80]},
    "ema_pull": {"ema_f": [5, 10, 20, 35, 50]},
    "momentum": {"mom_k": [6, 12, 24, 48, 72], "mom_x": [0.75, 1.0, 1.5, 2.0]},
}


def v1_baseline():
    path = os.path.join(RESULTS, "final_config.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)["params"]


# ---------------------------------------------------------------- runner
def phase(store, configs, name, seen=None):
    rows = []
    t0 = time.time()
    seen = seen if seen is not None else set()
    for i, p in enumerate(configs, 1):
        p = clean(p)
        key = json.dumps(p, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        _, tr, te = bt.run_config(store, p)
        rows.append({**p,
                     **{f"train_{k}": v for k, v in tr.items()},
                     **{f"test_{k}": v for k, v in te.items()}})
        if i % 50 == 0:
            print(f"  {name} {i}/{len(configs)}  {time.time()-t0:.0f}s",
                  flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(os.path.join(RESULTS, f"v2_{name}.csv"), index=False)
    print(f"  {name}: {len(df)} configs in {time.time()-t0:.0f}s", flush=True)
    return df


def train_rank(df):
    d = bt.rank(df, min_train_n=MIN_TRAIN_N, min_pf=MIN_TRAIN_PF)
    return d


# ---------------------------------------------------------------- stages
def stage_n1(store):
    cfgs = []
    base = v1_baseline()
    cfgs.append(dict(base))                      # forced v1 baseline
    for es in entry_sets():
        for filt in ("none", "ema200"):
            for tf in TFS:
                for ex in exit_presets():
                    p = dict(es)
                    p.update(ex)
                    p.update(dict(tf=tf, filter=filt,
                                  time_stop=bt.TSTOP[tf][1]))
                    cfgs.append(p)
    print(f"N1: {len(cfgs)} configs", flush=True)
    return phase(store, cfgs, "n1")


def top_parents(n1df, per_family=2, cap=18):
    """Best rows by train score, but with at least `per_family` parents per
    ENTRY FAMILY first (so one dominant family cannot crowd out the rest)."""
    d = train_rank(n1df)
    recs = d.to_dict("records")
    seen, parents, fam = set(), [], {}

    def add(r):
        k = json.dumps(clean(param_dict(r)), sort_keys=True)
        if k in seen:
            return
        seen.add(k)
        parents.append(r)

    for r in recs:                       # pass 1: family diversity
        if fam.get(r["entry"], 0) >= per_family:
            continue
        fam[r["entry"]] = fam.get(r["entry"], 0) + 1
        add(r)
    for r in recs:                       # pass 2: fill by score
        if len(parents) >= cap:
            break
        add(r)
    counts = {}
    for r in parents:
        counts[r["entry"]] = counts.get(r["entry"], 0) + 1
    print(f"N2 parents: {len(parents)} {counts}", flush=True)
    return parents


def fine_exits(par):
    """Exit grid appropriate to the parent's exit mode."""
    ts, tf = par["time_stop"], par["tf"]
    out = []
    if par["exit"] == "fixed":
        for tp, sl in itertools.product((2.0, 3.0, 4.0), (1.5, 2.0, 3.0)):
            out.append(dict(exit="fixed", tp_atr=tp, sl_atr=sl,
                            time_stop=ts))
    elif par["exit"] == "trail":
        for tp1, sl, frac, trl in itertools.product(
                (0.3, 0.5), (2.0, 4.0), (0.1, 0.4), (1.0, 2.0)):
            out.append(dict(exit="trail", tp1_atr=tp1, sl_atr=sl,
                            frac1=frac, tp2_atr=0.0, trail_atr=trl,
                            time_stop=ts))
    else:                                        # scale
        for tp1, tp2, sl in itertools.product(
                (0.5, 1.0), (2.0, 3.0, 5.0), (1.5, 2.5)):
            out.append(dict(exit="scale", tp1_atr=tp1, tp2_atr=tp2,
                            sl_atr=sl, frac1=0.5, time_stop=ts))
    return out


MODE_KEYS = ("bb_len", "bb_dev", "rsi_th", "dc_len", "ema_f",
             "mom_k", "mom_x")


def entry_neighbours(par, radius=1):
    dims = ENTRY_DIMS[par["entry"]]
    out = [dict(par)]
    for k, vals in dims.items():
        if k not in par:
            continue
        i = vals.index(par[k]) if par[k] in vals else None
        if i is None:
            cand = vals
        else:
            lo, hi = max(0, i - radius), min(len(vals), i + radius + 1)
            cand = vals[lo:hi]
        for v in cand:
            q = dict(par)
            q[k] = v
            out.append(q)
    # dedupe
    seen, res = set(), []
    for q in out:
        key = json.dumps(q, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            res.append(q)
    return res


def strip(rec, keys):
    return {k: rec[k] for k in keys if k in rec}


def stage_n2(store, n1df):
    parents = top_parents(n1df)
    mode_keys = [k for k in MODE_KEYS if any(k in p for p in parents)]
    base_keys = ["entry", "tf", "filter"] + mode_keys

    # --- n2a: fine exits, parent entry/filter/tf frozen
    seen = set()
    cfgs = []
    for par in parents:
        b = strip(par, base_keys)
        for ex in fine_exits(par):
            p = dict(b)
            p.update(ex)
            cfgs.append(p)
    n2a = phase(store, cfgs, "n2a", seen)

    # --- n2b: entry neighbours of best N2a rows (exit/filter kept)
    d = train_rank(n2a)
    best = d.head(10).to_dict("records")
    cfgs = []
    for par in best:
        for q in entry_neighbours(par, radius=1):
            cfgs.append(q)
    n2b = phase(store, cfgs, "n2b", seen)

    # --- n2c: time-stop dimension on the best of n2a+n2b
    both = pd.concat([n2a, n2b], ignore_index=True)
    d2 = train_rank(both)
    cfgs = []
    for par in d2.head(10).to_dict("records"):
        for ts in bt.TSTOP[par["tf"]]:
            if ts != par["time_stop"]:
                q = dict(par)
                q["time_stop"] = ts
                cfgs.append(q)
    n2c = phase(store, cfgs, "n2c", seen)
    return seen


# --------------------------------------------------------------- finals
def collect(seen=None):
    frames = []
    for name in ("n1", "n2a", "n2b", "n2c"):
        p = os.path.join(RESULTS, f"v2_{name}.csv")
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=[c for c in df.columns
                                    if c != "score"])
    return df


def picks_on_train(df):
    f = df[(df.train_n >= MIN_TRAIN_N) & (df.train_pf >= MIN_TRAIN_PF)
           & np.isfinite(df.train_rr)].copy()
    f["score"] = f.train_win_rate + 5.0 * np.minimum(f.train_pf, 3.0)
    out = {}
    a = f[f.train_win_rate >= 80.0]
    out["A"] = (a.sort_values(["train_rr", "score"], ascending=False)
                 .iloc[0].to_dict() if len(a) else None)
    b = f[f.train_rr >= 1.5]
    out["B"] = (b.sort_values(["train_win_rate", "score"], ascending=False)
                .iloc[0].to_dict() if len(b) else None)
    out["C"] = (f.sort_values("score", ascending=False).iloc[0].to_dict()
                if len(f) else None)
    return out, len(f)


def param_dict(row):
    keys = ["tf", "entry", "bb_len", "bb_dev", "rsi_th", "filter", "exit",
            "tp_atr", "sl_atr", "tp1_atr", "tp2_atr", "frac1", "trail_atr",
            "be_atr", "dc_len", "ema_f", "mom_k", "mom_x", "time_stop"]
    d = {k: row[k] for k in keys if k in row}
    return clean(d)


def reeval(store, p):
    trades, tr, te = bt.run_config(store, p)
    return trades, tr, te


def subperiods(trades):
    """Train-only robustness: WR/PF per era."""
    eras = [("2018-2020", "2018-01-01", "2021-01-01"),
            ("2021-2022", "2021-01-01", "2023-01-01"),
            ("2023-2024", "2023-01-01", "2025-01-01")]
    out = []
    for name, a, b in eras:
        sub = [t for t in trades
               if a <= str(t["entry_time"])[:10] < b]
        m = bt.metrics(sub)
        out.append((name, m["n"], m["win_rate"], m["pf"]))
    return out


def stage_final(store):
    df = collect()
    total_screened = len(df)
    pks, n_eligible = picks_on_train(df)
    base = v1_baseline()

    print(f"\ntotal distinct configs screened: {total_screened} "
          f"(eligible under filters: {n_eligible})")

    results = {}
    labels = {"A": "Track A -- WR>=80, max RR (train)",
              "B": "Track B -- RR>=1.5, max WR (train, primary)",
              "C": "Track C -- max score (train)"}
    for k, row in pks.items():
        if row is None:
            print(f"{labels[k]}: NO candidate found")
            continue
        p = param_dict(row)
        trades, tr, te = reeval(store, p)
        sp = subperiods(trades)
        results[k] = dict(params=p, train=tr, test=te, sub=sp)
        print(f"\n{labels[k]}")
        print(f"  params: {json.dumps(p, default=str)}")
        print(f"  train: WR={tr['win_rate']:.1f} RR={tr['rr']:.2f} "
              f"PF={tr['pf']:.2f} exp={tr['expectancy']:.3f}% n={tr['n']}")
        print(f"  test : WR={te['win_rate']:.1f} RR={te['rr']:.2f} "
              f"PF={te['pf']:.2f} exp={te['expectancy']:.3f}% n={te['n']}")
        print("  train eras: " + "; ".join(
            f"{nm}: n={n} WR={w:.1f} PF={pf:.2f}"
            for nm, n, w, pf in sp))

    # forced baseline re-scored on the same extended data
    b_trades, b_tr, b_te = reeval(store, base)
    b_sp = subperiods(b_trades)
    results["V1"] = dict(params=base, train=b_tr, test=b_te, sub=b_sp)
    print("\nV1 baseline (forced, re-scored on extended data)")
    print(f"  train: WR={b_tr['win_rate']:.1f} RR={b_tr['rr']:.2f} "
          f"PF={b_tr['pf']:.2f} exp={b_tr['expectancy']:.3f}% n={b_tr['n']}")
    print(f"  test : WR={b_te['win_rate']:.1f} RR={b_te['rr']:.2f} "
          f"PF={b_te['pf']:.2f} exp={b_te['expectancy']:.3f}% n={b_te['n']}")

    meta = dict(screened=total_screened, eligible=n_eligible,
                min_train_n=MIN_TRAIN_N, min_train_pf=MIN_TRAIN_PF,
                train_end=str(bt.TRAIN_END.date()))
    with open(os.path.join(RESULTS, "best_results.json"), "w",
              encoding="utf-8") as f:
        json.dump({"meta": meta, "picks": results}, f, ensure_ascii=False,
                  indent=2, default=str)
    _write_report(results, meta, df)
    print("\nwrote results/best_results.json, results/best_report.md")


def _row_md(m):
    return (f"{m['win_rate']:.1f} | {m['rr']:.2f} | {m['pf']:.2f} | "
            f"{m['expectancy']:.3f} | {m['n']} | {m['max_dd']:.1f}")


def _write_report(res, meta, df):
    L = ["# گزارش جستجوی بهترین استراتژی (دادهٔ توسعه‌یافته ۲۰۱۸–۲۰۲۶، ۴۲ نماد)\n"]
    L.append(f"- کل ترکیب‌های آزموده‌شده: **{meta['screened']}** "
             f"(فیلتر شرط: train_n≥{meta['min_train_n']}، "
             f"PF≥{meta['min_train_pf']} → {meta['eligible']} واجد شرایط)")
    L.append(f"- تمام انتخاب‌ها فقط با دادهٔ آموزشی (تا {meta['train_end']})؛ "
             "تست ۲۰۲۵–۲۰۲۶ فقط گزارش می‌شود")
    L.append("- صداقت: پنجرهٔ ۲۰۲۵+ در نسخهٔ قبل دیده شده؛ اعداد تستِ این "
             "پیکربندی‌های نو «اولین نگاه به همین ترکیب دقیق» است، نه دادهٔ "
             "کاملاً ناآشنا\n")
    hdr = ["پیکربندی", "دوره", "WR٪", "R:R", "PF", "انتظار٪", "n",
           "حداکثر افت٪"]
    L.append("| " + " | ".join(hdr) + " |")
    L.append("|" + "---|" * len(hdr))
    order = [("B", "🏆 اصلی (R:R≥۱٫۵، حداکثر WR)"),
             ("A", "بالاترین WR (≥۸۰٪)"),
             ("C", "بهترین امتیاز ترکیبی"),
             ("V1", "نسخهٔ قبل (baseline)")]
    for k, label in order:
        if k not in res:
            continue
        for per, m in (("آموزشی", res[k]["train"]), ("آزمونی", res[k]["test"])):
            L.append(f"| {label if per == 'آموزشی' else ''} | {per} | "
                     + _row_md(m) + " |")
    L.append("")
    L.append("## پایداری آموزشی به تفکیک دوره (فقط دادهٔ آموزشی)\n")
    L.append("| پیکربندی | ۲۰۱۸–۲۰۲۰ | ۲۰۲۱–۲۰۲۲ | ۲۰۲۳–۲۰۲۴ |")
    L.append("|---|---|---|---|")
    for k, label in order:
        if k not in res:
            continue
        cells = " | ".join(
            f"n={n}, WR={w:.1f}%, PF={pf:.2f}"
            for _, n, w, pf in res[k]["sub"])
        L.append(f"| {label} | {cells} |")
    L.append("")
    L.append("## پیکربندی‌های منتخب (JSON: results/best_results.json)\n")
    for k, label in order:
        if k in res:
            L.append(f"- **{label}**: `{json.dumps(res[k]['params'], default=str)}`")
    L.append("")
    L.append("## نتیجهٔ صادقانه\n")
    L.append("- این اعداد خروجی واقعی موتور بک‌تست روی دادهٔ واقعی Binance "
             "هستند؛ هیچ عددی دستکاری نشده.")
    L.append("- بک‌تست گذشته تضمین آینده نیست؛ بازار ۲۰۲۵–۲۰۲۶ با ۲۰۱۸–۲۰۲۲ "
             "متفاوت است و ستون‌های دوره‌ای همین را نشان می‌دهند.")
    L.append("- برای معاملهٔ واقعی ابتدا چند هفته paper-test با `monitor.py` "
             "انجام دهید.")
    with open(os.path.join(RESULTS, "best_report.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


# ------------------------------------------------------------------ main
def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    store = build_store()
    if what in ("n1", "all"):
        stage_n1(store)
    if what in ("n2", "all"):
        n1df = pd.read_csv(os.path.join(RESULTS, "v2_n1.csv"))
        stage_n2(store, n1df)
    if what in ("final", "all"):
        stage_final(store)


if __name__ == "__main__":
    main()
