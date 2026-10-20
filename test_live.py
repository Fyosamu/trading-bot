"""
End-to-end tests of the LIVE pipeline.  Nothing in data/ is modified.

T1  data parity     : live klines from data-api.binance.vision vs the local
                      archive, on every overlapping bar (max relative OHLC diff)
T2  signal parity   : build_dirs() fed IDENTICAL overlapping windows from both
                      sources must return identical sides
T3  monitor parity  : the monitor's real path (1000-bar fetch) vs the
                      full-history backtest path -- the side of the LAST common
                      bar (what actually gets sent) must match
T4  news calendar   : Forex Factory JSON fetch + parse
T5  dedupe          : monitor state suppresses a second send for same key
T6  notify routing  : 'off' / 'not-configured' statuses without any network call

usage:  python test_live.py
exit 0 = all tests passed, exit 1 = at least one failure
"""
import json
import sys

import pandas as pd

import backtest as bt
import monitor
import notify

RESULTS = []
SKIPS = []


def check(name, ok, info=""):
    RESULTS.append((name, bool(ok), info))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {info}" if info else ""))


def skip(name, info):
    SKIPS.append((name, info))
    print(f"[SKIP] {name}  -- {info}")


def to_sd(df):
    if "time" in df.columns:
        t = df["time"].to_numpy()
    else:                       # df is indexed by bar time
        t = pd.DatetimeIndex(df.index).to_numpy()
    return bt._make_sd(t, df["open"].to_numpy(),
                       df["high"].to_numpy(), df["low"].to_numpy(),
                       df["close"].to_numpy(), df["volume"].to_numpy())


def main():
    with open(monitor.CONFIG_PATH, encoding="utf-8") as f:
        p = json.load(f)["params"]

    # ---------------------------------------------------------------- T4
    try:
        ev = monitor.upcoming_news()
        check("T4 news calendar fetch+parse", True,
              f"USD High/Med events in next 60min: {len(ev)}")
    except Exception as e:
        check("T4 news calendar fetch+parse", False, str(e))

    # ---------------------------------------------------------------- T5
    st = {"seen": []}
    key = "sig:TEST:4h:2026-01-01 00:00:00:1"
    first, second = monitor._seen(st, key), monitor._seen(st, key)
    check("T5 dedupe suppresses repeat", (not first) and second,
          "first send -> allowed, second -> suppressed")

    # ---------------------------------------------------------------- T6
    cfg = {
        "ntfy": {"enabled": False, "topic": "t"},
        "telegram": {"enabled": True, "bot_token": "", "chat_id": ""},
        "discord": {"enabled": False, "webhook_url": ""},
        "email": {"enabled": False, "host": "", "port": 587,
                  "user": "", "password": "", "to": ""},
        "webhook": {"enabled": False, "url": ""},
    }
    res = notify.send("routing test", "no network should be used here", cfg)
    ok = (res.get("ntfy") == "off"
          and res.get("telegram") == "not-configured"
          and res.get("discord") == "off"
          and res.get("email") == "off"
          and res.get("webhook") == "off")
    check("T6 notify routing (offline statuses)", ok, str(res))

    # ---------------------------------------------------------- T1/T2/T3
    n_t2 = n_t3 = 0
    live_sides = []
    for sym in bt.SYMBOLS:
        try:
            live = monitor.fetch_closed_klines(sym, p["tf"], limit=1000)
            loc = bt.load_symbol(sym, p["tf"])
        except Exception as e:
            check(f"T1 {sym}", False, f"load failed: {e}")
            continue

        lvi = live.set_index("time")
        lts = pd.DatetimeIndex(loc["time"])
        common = lvi.index.intersection(lts).sort_values()
        if len(common) == 0:
            skip(f"T1 {sym}", "archive and live have no overlapping bars")
            continue

        a = lvi.loc[common]
        b = pd.DataFrame(
            {k: loc[k] for k in ("open", "high", "low", "close", "volume")},
            index=lts).loc[common]

        # ---- T1: raw data parity on the overlap
        # threshold 1e-3 (0.1%): Binance's own live API and its monthly
        # archive can differ by a few ticks on individual bars (observed:
        # KAVAUSDT 2026-07-12 open, 2 ticks) -- T2 below is the strict,
        # zero-tolerance guarantee that no signal is affected.
        max_rel = 0.0
        for k in ("open", "high", "low", "close"):
            d = (a[k] - b[k]).abs() / b[k].abs().clip(lower=1e-9)
            max_rel = max(max_rel, float(d.max()))
        check(f"T1 data parity {sym}", max_rel <= 1e-3,
              f"bars={len(common)} max_rel_diff={max_rel:.2e}")

        # ---- T2: identical window -> identical signals
        if len(common) >= 50:
            dA = bt.build_dirs(to_sd(a), p)
            dB = bt.build_dirs(to_sd(b), p)
            mm = int((dA != dB).sum())
            check(f"T2 same-window signals {sym}", mm == 0,
                  f"bars={len(common)} mismatched_bars={mm}")
            n_t2 += 1
        else:
            skip(f"T2 {sym}", f"overlap {len(common)} bars < 50")

        # ---- T3: monitor's real path vs full-history path
        dirs_live = bt.build_dirs(to_sd(live), p)
        dirs_full = bt.build_dirs(loc, p)
        last = common[-1]
        posL = int(lvi.index.get_loc(last))
        posF = int(lts.get_loc(last))
        sideL, sideF = int(dirs_live[posL]), int(dirs_full[posF])
        k10 = common[-min(10, len(common)):]
        flips = sum(int(dirs_live[int(lvi.index.get_loc(t))])
                    != int(dirs_full[int(lts.get_loc(t))]) for t in k10)
        check(f"T3 monitor-vs-fullhistory last bar {sym}", sideL == sideF,
              f"live={sideL:+d} full={sideF:+d} flips_last10={flips}/10")
        n_t3 += 1
        if sideL != 0:
            live_sides.append(f"{sym}:{sideL:+d}")

    if live_sides:
        print(f"[INFO] live signals on the last closed {p['tf']} bar: "
              + ", ".join(live_sides))
    else:
        print(f"[INFO] no live signals on the last closed {p['tf']} bar "
              "(normal -- not every bar fires)")

    # ------------------------------------------------------------- summary
    fails = [r for r in RESULTS if not r[1]]
    print(f"\n==== {len(RESULTS) - len(fails)}/{len(RESULTS)} PASS, "
          f"{len(fails)} FAIL, {len(SKIPS)} SKIP (T2 on {n_t2} syms, "
          f"T3 on {n_t3} syms) ====")
    for name, _, info in fails:
        print(f"  FAILED: {name} -- {info}")
    for name, info in SKIPS:
        print(f"  skipped: {name} -- {info}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
