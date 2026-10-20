"""
News-direction backtest: can we predict the NEXT H-hour direction of the
market at high-impact news moments (FOMC / CPI / NFP)?

Protocol (honest, same as the strategy backtest):
  * decision bar: close of the 2nd 1h bar after the bar containing the news
    (CPI/NFP 08:30 -> decision 10:00 ET; FOMC 14:00 -> 16:00 ET)
  * entry = decision close; label = sign(close[H bars later] - entry)
  * ties (flat move) are dropped, not counted
  * ALL rule/horizon/group combos are scored; the winner is chosen on TRAIN
    (events before 2025-01-01) and only then its TEST number is shown
  * sample = (event x symbol); symbols are correlated inside one event, so
    the report shows both n_samples and n_events -- do not read n_samples
    as independent trials

usage:  python news_backtest.py
"""
import itertools
import os

import numpy as np
import pandas as pd

import backtest as bt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
EVENTS = os.path.join(HERE, "data", "news_events.csv")

HORIZONS = (1, 2, 4, 8)
GROUPS = ("ALL", "FOMC", "CPI", "NFP")
MIN_TRAIN_N = 300


# ------------------------------------------------------------------ features
def feature_rows(sd, ev, sym):
    """Rows for one symbol x all events. ev = DataFrame(name,kind,time_utc)."""
    t = sd["time"]
    c = sd["close"]
    hi = sd["high"]
    lo = sd["low"]
    atr = sd["atr"]
    n = len(c)
    rows = []
    for e in ev.itertuples():
        t0 = np.datetime64(pd.Timestamp(e.time_utc).floor("h"))
        i = int(np.searchsorted(t, t0))
        if i >= n or t[i] != t0:
            continue                      # no bar at that hour (data gap)
        if i < 40 or i + 1 + max(HORIZONS) >= n:
            continue
        c0, c1 = float(c[i]), float(c[i + 1])
        c_m1 = float(c[i - 1])
        a = float(atr[i + 1])
        if not a > 0:
            continue
        hi6 = float(hi[i - 6:i].max())
        lo6 = float(lo[i - 6:i].min())
        span = hi6 - lo6

        def sgn(x):
            return 1 if x > 0 else (-1 if x < 0 else 0)

        rng_pos = (c1 - lo6) / span if span > 0 else 0.5
        rows.append(dict(
            symbol=sym, kind=e.kind, when=e.time_utc,
            dir_post=sgn(c1 - c0),
            dir_full=sgn(c1 - c_m1),
            pre4=sgn(c0 - float(c[i - 4])),
            pre8=sgn(c0 - float(c[i - 8])),
            pre24=sgn(c0 - float(c[i - 24])),
            brk=(1 if c1 > hi6 else (-1 if c1 < lo6 else 0)),
            react=abs(c1 - c0) / a,
            rng_pos=rng_pos,
            entry=c1,
            **{f"y{h}": sgn(float(c[i + 1 + h]) - c1) for h in HORIZONS},
        ))
    return rows


# -------------------------------------------------------------------- rules
def rules(r):
    """name -> side in {-1,0,+1}; 0 = no trade (skip)."""
    post, full, p8, p24 = r.dir_post, r.dir_full, r.pre8, r.pre24
    out = {
        "post": post,
        "full": full,
        "pre4": r.pre4,
        "pre8": p8,
        "pre24": p24,
        "agree8": post if post == p8 else 0,
        "agree24": post if post == p24 else 0,
        "brk": r.brk,
        "brk_agree": r.brk if r.brk == post and r.brk != 0 else 0,
        "fade_rng": (1 if r.rng_pos <= 0.15 else
                     (-1 if r.rng_pos >= 0.85 else 0)),
    }
    for thr in (0.3, 0.6, 1.0):
        out[f"post_r{thr}"] = post if r.react >= thr else 0
    for thr in (0.6, 1.0):
        out[f"fade_r{thr}"] = -post if r.react >= thr else 0
    return out


def evaluate(df):
    """Score every (group, rule, horizon) combo; return tidy frame."""
    df = df.copy()
    df["when"] = pd.to_datetime(df["when"])
    df["train"] = df["when"] < bt.TRAIN_END
    scores = []
    for g in GROUPS:
        sub = df if g == "ALL" else df[df.kind == g]
        if sub.empty:
            continue
        # precompute rule sides once per row
        sides = pd.DataFrame([rules(r) for r in sub.itertuples()],
                             index=sub.index)
        yr = sub["when"].dt.year
        for h in HORIZONS:
            y = sub[f"y{h}"]
            for name in sides.columns:
                side = sides[name]
                ok = (side != 0) & (y != 0)   # drop no-trade and flat ties
                res = (side == y)          # correctness
                tr = sub.train & ok
                te = (~sub.train) & ok
                ntr, nte = int(tr.sum()), int(te.sum())
                if ntr < MIN_TRAIN_N:
                    continue
                w23 = res[tr & (yr == 2023)]
                w24 = res[tr & (yr == 2024)]
                scores.append(dict(
                    group=g, rule=name, H=h,
                    n_tr=ntr, ev_tr=int(sub.loc[tr, "when"].nunique()),
                    wr_tr=100.0 * float(res[tr].mean()),
                    wr_23=(100.0 * float(w23.mean())
                           if len(w23) >= 80 else np.nan),
                    wr_24=(100.0 * float(w24.mean())
                           if len(w24) >= 80 else np.nan),
                    n_te=nte, ev_te=int(sub.loc[te, "when"].nunique()),
                    wr_te=100.0 * float(res[te].mean()) if nte else np.nan,
                ))
    return pd.DataFrame(scores).sort_values(
        ["wr_tr", "n_tr"], ascending=False).reset_index(drop=True)


def select_stable(grid):
    """Selection rule fixed in advance, TRAIN data only:
    at least 20 events in train, then maximize the WORST of the 2023 / 2024
    win rates (regime-robust), so a single lucky year cannot win."""
    c = grid[(grid.ev_tr >= 20) & grid.wr_23.notna()
             & grid.wr_24.notna()].copy()
    if c.empty:
        return grid.iloc[0]
    c["worst"] = c[["wr_23", "wr_24"]].min(axis=1)
    return c.sort_values(["worst", "wr_tr"],
                         ascending=False).iloc[0]


def main():
    ev = pd.read_csv(EVENTS, parse_dates=["time_utc"])
    print(f"events: {len(ev)}  ({ev.kind.value_counts().to_dict()})")

    rows = []
    for sym in bt.SYMBOLS:
        try:
            sd = bt.load_symbol(sym, "1h")
        except Exception as e:
            print(f"  {sym}: load failed ({e})")
            continue
        rows.extend(feature_rows(sd, ev, sym))
    df = pd.DataFrame(rows)
    print(f"sample rows (event x symbol): {len(df)}  "
          f"train events before {bt.TRAIN_END.date()}")

    grid = evaluate(df)
    os.makedirs(RESULTS, exist_ok=True)
    grid.to_csv(os.path.join(RESULTS, "news_grid.csv"), index=False)

    cols = ["group", "rule", "H", "n_tr", "ev_tr", "wr_tr",
            "n_te", "ev_te", "wr_te"]
    print(f"\nconfigs scored: {len(grid)} (train n >= {MIN_TRAIN_N})")
    print("\n=== TOP 15 by TRAIN win-rate ===")
    print(grid[cols].head(15).to_string(index=False))

    best = select_stable(grid)
    print(f"\n>>> chosen on train (max WORST-year WR, >=20 events): "
          f"{best.group}/{best.rule} horizon {int(best.H)}h")
    print(f"    train: WR={best.wr_tr:.1f}%  2023={best.wr_23:.1f}%  "
          f"2024={best.wr_24:.1f}%  n={int(best.n_tr)} "
          f"({int(best.ev_tr)} events)")
    print(f"    test : WR={best.wr_te:.1f}%  n={int(best.n_te)} "
          f"({int(best.ev_te)} events)")

    # honest ceiling disclosure: best TEST number (not selectable)
    bt_best = grid[grid.n_te >= 200].sort_values("wr_te", ascending=False)
    if not bt_best.empty:
        b = bt_best.iloc[0]
        print(f"\n(info) best TEST value in grid -- NOT selectable: "
              f"{b.group}/{b.rule} H={int(b.H)} test WR={b.wr_te:.1f}%")

    _write_report(df, grid, best)
    print(f"\nwrote results/news_grid.csv, results/news_report.md")


def _write_report(df, grid, best):
    L = []
    L.append("# گزارش بک‌تست جهت بازار در تایم خبر (FOMC / CPI / NFP)\n")
    L.append(f"- رویدادها: {len(pd.read_csv(EVENTS))} عدد از منابع رسمی "
             "فدرال رزرو و BLS (۲۰۲۳ تا ۲۰۲۶)")
    L.append(f"- نمونه‌ها: {len(df)} (رویداد × نماد)؛ نمادها داخل یک رویداد "
             "هم‌بسته‌اند — بنابراین تعداد «رویداد» ستون مهم‌تری است")
    L.append(f"- انتخاب قاعده فقط با آموزشی (قبل از {bt.TRAIN_END.date()})؛ "
             "ستون WR آزمونی فقط گزارش می‌شود\n")
    L.append("## ۲۰ قاعده برتر بر اساس Win Rate آموزشی\n")
    cols = ["group", "rule", "H", "n_tr", "ev_tr", "wr_tr", "n_te",
            "ev_te", "wr_te"]
    hdr = ["گروه", "قاعده", "افق(h)", "n آموزشی", "رویداد", "WR آموزشی",
           "n آزمونی", "رویداد", "WR آزمونی"]
    L.append("| " + " | ".join(hdr) + " |")
    L.append("|" + "---|" * len(hdr))
    for r in grid[cols].head(20).itertuples(index=False):
        te = "—" if np.isnan(r.wr_te) else f"{r.wr_te:.1f}%"
        L.append(f"| {r.group} | {r.rule} | {int(r.H)} | {int(r.n_tr)} | "
                 f"{int(r.ev_tr)} | {r.wr_tr:.1f}% | {int(r.n_te)} | "
                 f"{int(r.ev_te)} | {te} |")
    L.append(f"\n**قاعده انتخاب‌شده (فقط با آموزشی، حداکثر WR بدترین سال "
             f"آموزشی):** `{best.group}/{best.rule}` افق {int(best.H)} ساعت — "
             f"WR آموزشی {best.wr_tr:.1f}% (۲۰۲۳: {best.wr_23:.1f}%، "
             f"۲۰۲۴: {best.wr_24:.1f}%)، WR آزمونی {best.wr_te:.1f}%\n")
    L.append("## هشدارهای صادقانه\n")
    L.append("- هدف ۷۵٪ وین‌ریت در پیش‌بینی جهت خبری با این قواعد روی این "
             "داده به‌دست نیامد؛ بالاترین WR واقعی همین عدد بالاست و در "
             "ستون‌ها دیده می‌شود. هر عدد بالاتری ادعای جعلی است.")
    L.append("- تعداد زیادی ترکیب (قاعده × افق × گروه) آزموده شده؛ انتخاب "
             "روی آموزشی انجام شده تا WR آزمونی بی‌طرف بماند.")
    L.append("- نمونه‌های یک رویداد روی ۲۷ نماد هم‌جهت هستند (کریپتو در "
             "خبر ماکرو یکجا حرکت می‌کند) — آمار مستقل ≈ تعداد رویداد است.")
    L.append("- پیشنهاد عملی: از این مدل برای «جهت‌گیری احتمالی» استفاده "
             "کنید نه سیگنال قطعی؛ برای ورود از ربات اصلی (BB+RSI) کمک "
             "بگیرید و در پنجره خبر حجم را کم کنید.")
    with open(os.path.join(RESULTS, "news_report.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
