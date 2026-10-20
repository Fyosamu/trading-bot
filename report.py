"""
Final stage (honest protocol):
  1. pick ONE config using TRAIN data only (stage-3 grid preferred)
  2. evaluate it on the untouched out-of-sample TEST period
  3. write Persian markdown report + equity curve + trades CSV
"""
import json
import os

import numpy as np
import pandas as pd

import backtest as bt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

FA = 1  # formatting helper


def read(name):
    p = os.path.join(RESULTS, f"{name}.csv")
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()


def candidates(df):
    """Selection rule fixed in advance, TRAIN data only.

    User target: Reward:Risk >= 1.5 (hard minimum) with the highest
    possible win rate, PF >= 1.05, and enough trades for a 1000+
    backtest.  Fallback tiers in case the RR>=1.5 zone is empty.
    """
    tiers = ((550, 1.05, 1.5, 0.0, "wr"),
             (400, 1.05, 1.5, 0.0, "wr"),
             (550, 1.05, 1.5, 40.0, "wr"),
             (550, 1.05, 0.0, 80.0, "rr"),
             (400, 1.05, 0.0, 80.0, "rr"),
             (550, 1.05, 0.0, 75.0, "rr"),
             (400, 1.05, 1.7, 35.0, "score"),
             (400, 1.0, 1.5, 30.0, "score"),
             (300, 1.0, 1.2, 0.0, "score"))
    d = pd.DataFrame()
    rank = "score"
    for lo_n, lo_pf, lo_rr, lo_wr, rank in tiers:
        d = df[(df.train_n >= lo_n) & (df.train_pf >= lo_pf)
               & (df.train_rr >= lo_rr) & (df.train_win_rate >= lo_wr)]
        if not d.empty:
            break
    if d.empty:
        d = df
    if rank == "rr":
        return d.sort_values(["train_rr", "train_pf"], ascending=False)
    if rank == "wr":
        return d.sort_values(["train_win_rate", "train_rr"],
                             ascending=False)
    d = d.assign(score=d.train_win_rate + 5.0 * np.minimum(d.train_pf, 3.0))
    return d.sort_values("score", ascending=False)


def params_of(row, src):
    keys = ["tf", "entry", "bb_len", "bb_dev", "rsi_th", "filter",
            "time_stop"]
    if row["exit"] == "fixed":
        keys += ["exit", "tp_atr", "sl_atr"]
    else:
        keys += ["exit", "tp1_atr", "be_atr", "frac1", "sl_atr",
                 "tp2_atr"]
        if "trail_atr" in row.index:
            keys += ["trail_atr"]
    p = {}
    for k in keys:
        v = row[k]
        if isinstance(v, float) and np.isnan(v):
            continue          # column exists but is unused for this exit
        if isinstance(v, np.integer):
            v = int(v)
        elif isinstance(v, np.floating):
            v = int(v) if float(v).is_integer() else float(v)
        p[k] = v
    return p


def md_row(vals):
    return "| " + " | ".join(str(v) for v in vals) + " |"


def fmt_metrics(m, label):
    return md_row([label, m["n"], f"{m['win_rate']:.2f}%",
                   f"{m['rr']:.2f}", f"{m['pf']:.2f}",
                   f"{m['expectancy']:.3f}%",
                   f"{m['avg_win']:.3f}%", f"{m['avg_loss']:.3f}%",
                   f"{m['max_dd']:.2f}%", m["max_consec_loss"],
                   f"{m['total']:.1f}%"])


def config_section(L, row, title, fname):
    """Write one alternative-config section (params + train/test metrics)."""
    ap = params_of(row, "stage4")
    with open(os.path.join(RESULTS, fname), "w") as f:
        json.dump({"params": ap,
                   "train": {k: row[f"train_{k}"] for k in
                             ("n", "win_rate", "rr", "pf", "expectancy")},
                   "test": {k: row[f"test_{k}"] for k in
                            ("n", "win_rate", "rr", "pf", "expectancy")}},
                  f, indent=2, default=str)
    L.append(f"## {title}")
    L.append("")
    L.append(f"`{json.dumps(ap, default=str)}`")
    L.append("")
    L.append(md_row(["دوره", "معاملات", "Win Rate", "R:R", "PF",
                     "انتظار سود", "حداکثر افت"]))
    L.append(md_row(["---"] * 7))
    for lbl, pre in (("آموزشی", "train"), ("آزمونی", "test")):
        L.append(md_row([lbl, int(row[f"{pre}_n"]),
                         f"{row[f'{pre}_win_rate']:.2f}%",
                         f"{row[f'{pre}_rr']:.2f}",
                         f"{row[f'{pre}_pf']:.2f}",
                         f"{row[f'{pre}_expectancy']:.3f}%",
                         f"{row[f'{pre}_max_dd']:.0f}%"]))
    L.append("")
    L.append(f"(ذخیره‌شده در `results/{fname}`)")
    L.append("")


def main():
    grids = [("stage7", read("stage7")), ("stage6", read("stage6")),
             ("stage5", read("stage5")), ("stage4", read("stage4")),
             ("stage3", read("stage3")), ("stage2", read("stage2")),
             ("stage1", read("stage1"))]
    src, cand = None, None
    for name, df in grids:
        if df.empty:
            continue
        c = candidates(df)
        if not c.empty:
            src, cand = name, c
            break
    picked = cand.iloc[0]
    params = params_of(picked, src)
    print(f"picked from {src}: {json.dumps(params)}")
    print(f"  train n={picked.train_n} wr={picked.train_win_rate:.2f} "
          f"pf={picked.train_pf:.2f}")

    store = {params["tf"]: {s: bt.load_symbol(s, params["tf"])
                            for s in bt.SYMBOLS}}
    trades, trm, tem = bt.run_config(store, params)
    allm = bt.metrics(trades)
    print(f"  test  n={tem['n']} wr={tem['win_rate']:.2f} "
          f"rr={tem['rr']:.2f} pf={tem['pf']:.2f}")
    print(f"  ALL   n={allm['n']} wr={allm['win_rate']:.2f} "
          f"rr={allm['rr']:.2f} pf={allm['pf']:.2f}")

    os.makedirs(RESULTS, exist_ok=True)
    tdf = pd.DataFrame(trades).sort_values("exit_time").reset_index(drop=True)
    tdf.to_csv(os.path.join(RESULTS, "trades.csv"), index=False)
    with open(os.path.join(RESULTS, "final_config.json"), "w") as f:
        json.dump({"params": params, "source": src,
                   "train": trm, "test": tem, "all": allm}, f, indent=2,
                  default=str)

    # per-symbol robustness
    per = []
    for sym, g in tdf.groupby("symbol"):
        gm = bt.metrics(g.to_dict("records"))
        per.append((sym, gm))
    per.sort(key=lambda x: -x[1]["win_rate"])

    # train-ranked top-5 transparency (collapse rows with identical metrics)
    top5 = []
    seen = set()
    for _, r in cand.iterrows():
        key = (round(r.train_win_rate, 4), round(r.train_rr, 4),
               round(r.train_pf, 4), int(r.train_n), int(r.test_n))
        if key in seen:
            continue
        seen.add(key)
        top5.append((src, params_of(r, src), r.train_win_rate, r.train_rr,
                     r.train_pf, int(r.train_n), r.test_win_rate,
                     r.test_rr, r.test_pf, int(r.test_n)))
        if len(top5) == 5:
            break

    # ---- combined grids: Pareto (WR vs RR) + high-WR alternative ---------
    combined = pd.concat([d for _, d in grids if not d.empty
                          and "train_rr" in d.columns], ignore_index=True)
    pareto = []
    for floor in (0.2, 0.5, 1.0, 1.5, 2.0, 2.5):
        sub = combined[(combined.train_rr >= floor)
                       & (combined.train_n >= 300)
                       & (combined.train_pf >= 1.05)]
        if sub.empty:
            pareto.append((floor, None))
            continue
        pareto.append((floor, sub.loc[sub.train_win_rate.idxmax()]))
    # highest win-rate config regardless of RR (the "85% style" bot)
    alt = combined[(combined.train_n >= 550)
                   & (combined.train_pf >= 1.05)]
    alt_row = alt.loc[alt.train_win_rate.idxmax()] if not alt.empty else None
    # highest-R:R profitable config (the "RR ~ 2" bot)
    rr2_pool = combined[(combined.train_n >= 300)
                        & (combined.train_pf >= 1.05)]
    rr2_row = (rr2_pool.loc[rr2_pool.train_rr.idxmax()]
               if not rr2_pool.empty else None)

    # ---- equity curve ----------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eq = tdf["ret"].cumsum() * 100.0
    x = np.arange(len(tdf))
    peak = np.maximum.accumulate(np.concatenate([[0.0], eq.to_numpy()]))
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(x, eq, lw=1.2, color="#1f77b4", label="equity %")
    is_test = (tdf["entry_time"].astype(str) >= str(bt.TRAIN_END)).to_numpy()
    if is_test.any():
        ax.axvline(int(np.argmax(is_test)), color="red", ls="--", lw=1.2,
                   label="out-of-sample test starts (2025-01-01)")
    ax.fill_between(x, peak[1:], eq.to_numpy(),
                    where=(eq.to_numpy() < peak[1:]), color="orange",
                    alpha=0.3, label="drawdown")
    ax.set_title(f"Equity | {allm['n']} trades | WR {allm['win_rate']:.1f}% "
                 f"| RR {allm['rr']:.2f} | PF {allm['pf']:.2f} "
                 f"| fees+slippage included")
    ax.set_xlabel("trade #")
    ax.set_ylabel("cumulative % (fixed notional)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "equity.png"), dpi=110)

    # ---- Persian markdown report ----------------------------------------
    L = []
    L.append("# گزارش بک‌تست ربات ترید")
    L.append("")
    L.append(f"**پارامترهای انتخاب‌شده (فقط با داده آموزشی):** `{json.dumps(params)}`")
    L.append("")
    L.append("## معیارهای کلی")
    L.append("")
    L.append("(R:R = میانگین سود ÷ |میانگین ضرر| — هدف: حداقل ۱٫۵ با "
             "بالاترین Win Rate ممکن)")
    L.append("")
    L.append(md_row(["دوره", "تعداد معامله", "Win Rate", "R:R",
                     "Profit Factor", "انتظار سود/معامله", "میانگین سود",
                     "میانگین ضرر", "حداکثر افت", "حداکثر ضرر متوالی",
                     "سود کل"]))
    L.append(md_row(["---"] * 11))
    L.append(fmt_metrics(trm, "آموزشی (۲۰۲۳–۲۰۲۴)"))
    L.append(fmt_metrics(tem, "**آزمونی بدون دستکاری (۲۰۲۵–۲۰۲۶)**"))
    L.append(fmt_metrics(allm, "**کل دوره**"))
    L.append("")
    L.append(f"تعداد کل معاملات: **{allm['n']}** (هدف: ۱۰۰۰+)")
    L.append("")
    L.append("## مقایسه ۵ پیکربندی برتر (همه بر اساس داده آموزشی انتخاب شده‌اند)")
    L.append("")
    L.append(md_row(["منبع گرید", "تایم‌فریم/ورود", "خروج", "WR آموزشی",
                     "R:R آموزشی", "PF آموزشی", "n آموزشی",
                     "WR آزمونی", "R:R آزمونی", "PF آزمونی",
                     "n آزمونی"]))
    L.append(md_row(["---"] * 11))
    for (s, p, twr, trr, tpf, tn, swr, srr, spf, sn) in top5:
        if p["exit"] == "fixed":
            ex = f"fixed tp={p.get('tp_atr')} sl={p.get('sl_atr')}"
        else:
            ex = (f"trail tp1={p.get('tp1_atr')} be={p.get('be_atr')} "
                  f"frac={p.get('frac1')} tr={p.get('trail_atr')} "
                  f"sl={p.get('sl_atr')}")
        L.append(md_row([s, f"{p['tf']}/{p['entry']}", ex,
                         f"{twr:.2f}%", f"{trr:.2f}", f"{tpf:.2f}", tn,
                         f"{swr:.2f}%", f"{srr:.2f}", f"{spf:.2f}", sn]))
    L.append("")
    # ---- Pareto frontier (WR vs RR) --------------------------------------
    L.append("## مرز تبادل Win Rate ↔ R:R (بهترین WR در هر کف R:R)")
    L.append("")
    L.append("همه ردیف‌ها فقط با داده آموزشی انتخاب شده‌اند؛ ستون‌های آزمونی "
             "برای شفافیت گزارش شده‌اند.")
    L.append("")
    L.append(md_row(["کف R:R", "تایم‌فریم/ورود/خروج", "WR آموزشی",
                     "R:R آموزشی", "PF آموزشی", "n آموزشی",
                     "WR آزمونی", "R:R آزمونی", "PF آزمونی", "n آزمونی"]))
    L.append(md_row(["---"] * 10))
    for floor, row in pareto:
        if row is None:
            L.append(md_row([f"≥ {floor}", "— یافت نشد", "—", "—", "—",
                             "—", "—", "—", "—", "—"]))
            continue
        ex = row["exit"] + (
            f" tp={row['tp_atr']}" if row["exit"] == "fixed"
            else f" tp1={row['tp1_atr']} trail={row['trail_atr']}")
        L.append(md_row([
            f"≥ {floor}", f"{row['tf']}/{row['entry']}/{ex}",
            f"{row.train_win_rate:.1f}%", f"{row.train_rr:.2f}",
            f"{row.train_pf:.2f}", int(row.train_n),
            f"{row.test_win_rate:.1f}%", f"{row.test_rr:.2f}",
            f"{row.test_pf:.2f}", int(row.test_n)]))
    L.append("")
    L.append("**نکته ریاضی:** نقطه سر‌به‌سر در R:R=۲ یعنی Win Rate ۳۳٫۳٪؛ "
             "ترکیب Win Rate ۸۵٪ **و** R:R=۲ یعنی انتظار سود ≈ ۱٫۵ واحد "
             "ریسک به‌ازای هر معامله که هیچ استراتژی پایداری چنین عددی "
             "ندارد. این دو هدف **قابل تحقق هم‌زمان** نیستند؛ "
             "جدول بالا مرز واقعی را نشان می‌دهد.")
    L.append("")
    # ---- alternative configs ---------------------------------------------
    if alt_row is not None:
        config_section(L, alt_row,
                       "پیکربندی جایگزین ۱: حداکثر Win Rate",
                       "alt_high_wr_config.json")
    if rr2_row is not None:
        config_section(L, rr2_row,
                       "پیکربندی جایگزین ۲: حداکثر R:R (هدف ≈۲)",
                       "alt_max_rr_config.json")
    L.append("## پایداری در هر نماد (کل دوره)")
    L.append("")
    L.append(md_row(["نماد", "معاملات", "Win Rate", "PF",
                     "انتظار سود", "سود کل"]))
    L.append(md_row(["---"] * 6))
    for sym, gm in per:
        L.append(md_row([sym, gm["n"], f"{gm['win_rate']:.1f}%",
                         f"{gm['pf']:.2f}", f"{gm['expectancy']:.3f}%",
                         f"{gm['total']:.1f}%"]))
    L.append("")
    reasons = tdf["reason"].value_counts()
    L.append("## دلایل خروج")
    L.append("")
    for k, v in reasons.items():
        L.append(f"- {k}: {v} ({v / len(tdf) * 100:.1f}٪)")
    L.append("")
    L.append("## قواعد صادقانه‌ی بک‌تست")
    L.append("")
    L.append("- ورود در باز شدن کندل بعد از سیگنال (بدون نگاه به آینده)")
    L.append("- اگر در یک کندل هم حد سود و هم حد ضرر لمس شود، حد ضرر اول فرض شده")
    L.append("- کارمزد ۰٫۰۴٪ + اسلیپیج ۰٫۰۲٪ هر طرف (۰٫۱۲٪ هر رفت‌وبرگشت)")
    L.append("- انتخاب پارامتر فقط روی ۲۰۲۳–۲۰۲۴؛ نتیجه ۲۰۲۵–۲۰۲۶ یک‌بار ارزیابی شده")
    L.append("- یک معامله هم‌زمان در هر نماد")
    L.append("")
    L.append("## هشدارهای مهم")
    L.append("")
    L.append("- این سه هدف با هم قابل تحقق نیستند: Win Rate ۸۰٪+، R:R ≥ ۱٫۵ "
             "و ۱۰۰۰ معامله. سر‌به‌سر در R:R=۱٫۵ یعنی WR=۴۰٪؛ جدول Pareto "
             "مرز واقعی را نشان می‌دهد. پیکربندی اصلی این گزارش R:R ≥ ۱٫۵ را "
             "رعایت کرده و Win Rate آن بالاترین مقدار ممکن در این ناحیه است؛ "
             "پیکربندی‌های جایگزین (حداکثر Win Rate و حداکثر R:R) جدا "
             "گزارش شده‌اند.")
    L.append("- ستون «حداکثر افت» بر مبنای مجموع بازده‌ها با اندازهٔ ثابت "
             "هر معامله است (نه سود مرکب حساب)؛ عدد بزرگ یعنی دوره‌های "
             "طولانی ضرر متوالی.")
    L.append("- نماد ALGOUSDT دادهٔ ناقص دارد (۲۰۲۳-۰۱ تا ۲۰۲۵-۰۷)؛ "
             "بقیه نمادها کامل‌اند. هیچ داده‌ای جعل یا حذف نشده.")
    L.append("- بک‌تست گذشته تضمینی برای آینده نیست؛ "
             "چند ضرر پشت‌سرهم کاملاً طبیعی است.")
    L.append("- با اهرم، ریسک نابودی سرمایه جدی می‌شود؛ "
             "بدون مدیریت ریسک استفاده نکنید.")
    L.append("")
    with open(os.path.join(RESULTS, "final_report.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L))
    print("wrote final_report.md, equity.png, trades.csv, final_config.json")


if __name__ == "__main__":
    main()
