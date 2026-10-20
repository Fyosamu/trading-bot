"""
Live signal monitor.

Polls data-api.binance.vision (public REST, no API key) for the LAST
CLOSED candle of the strategy timeframe on every symbol, runs the SAME
bt.build_dirs() logic the backtest uses, and pushes a notification
through every enabled channel in notify_config.json.

Also sends pre-news alerts: High/Medium impact USD events from the free
Forex Factory JSON calendar, 0-60 minutes before release.

usage:
  python monitor.py --test    send one test message, print channel status
  python monitor.py --once    single pass now (signals + news), then exit
  python monitor.py           loop forever (default check every 5 min)

Dedup state: results/monitor_state.json  (one notification per
symbol/timeframe/candle -- never spams the same bar twice).
"""
import argparse
import datetime as dt
import json
import os
import time
import urllib.request

import pandas as pd

import backtest as bt
import notify

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
STATE_PATH = os.path.join(RESULTS, "monitor_state.json")
CONFIG_PATH = os.path.join(RESULTS, "final_config.json")

KLINES = ("https://data-api.binance.vision/api/v3/klines"
          "?symbol={s}&interval={i}&limit={n}")
FF_CAL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

CHECK_SEC = 300          # signal poll interval
NEWS_LOOKAHEAD_MIN = 60  # alert this many minutes before a news event
NEWS_POLL_SEC = 1800
STATE_KEEP = 800


# ------------------------------------------------------------------ state
def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"seen": []}


def save_state(st):
    os.makedirs(RESULTS, exist_ok=True)
    st["seen"] = st["seen"][-STATE_KEEP:]
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(st, f)


def _seen(st, key):
    if key in st["seen"]:
        return True
    st["seen"].append(key)
    return False


# ------------------------------------------------------------------ data
def fetch_closed_klines(sym, interval, limit=1000):
    """Last <=`limit` candles of `interval`, in-progress one dropped.
    1000 bars so EMA200/RSI/ATR are fully warmed up (at 400 bars the EMA
    still carries ~2% of its initial-condition error)."""
    url = KLINES.format(s=sym, i=interval, n=int(limit))
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        raw = json.loads(r.read())
    now_ms = int(time.time() * 1000)
    rows = [k for k in raw if int(k[6]) <= now_ms]   # drop open candle
    df = pd.DataFrame({
        "time": pd.to_datetime([int(k[0]) for k in rows], unit="ms"),
        "open": [float(k[1]) for k in rows],
        "high": [float(k[2]) for k in rows],
        "low": [float(k[3]) for k in rows],
        "close": [float(k[4]) for k in rows],
        "volume": [float(k[5]) for k in rows],
    })
    return df


# ---------------------------------------------------------------- signals
def check_signals(p):
    """Run the backtest's own signal logic on each symbol's last closed bar."""
    sigs = []
    for sym in bt.SYMBOLS:
        try:
            df = fetch_closed_klines(sym, p["tf"])
        except Exception as e:
            print(f"  {sym}: fetch failed ({e})")
            continue
        if len(df) < 250:
            continue
        sd = bt._make_sd(df["time"].to_numpy(),
                         df["open"].to_numpy(), df["high"].to_numpy(),
                         df["low"].to_numpy(), df["close"].to_numpy(),
                         df["volume"].to_numpy())
        side = int(bt.build_dirs(sd, p)[-1])
        if side == 0:
            continue
        bar = df.iloc[-1]
        a = float(sd["atr"][-1])
        c = float(bar["close"])
        sig = {"symbol": sym, "side": side, "bar": str(bar["time"]),
               "close": c, "atr": a}
        if p["exit"] == "fixed":
            sig["tp"] = c + side * p["tp_atr"] * a
            sig["sl"] = c - side * p["sl_atr"] * a
        else:
            sig["tp1"] = c + side * p["tp1_atr"] * a
            sig["sl"] = c - side * p["sl_atr"] * a
            if p.get("tp2_atr", 0):
                sig["tp2"] = c + side * p["tp2_atr"] * a
        sigs.append(sig)
    return sigs


MODE_KEYS = {"touch": ("bb_len", "bb_dev", "rsi_th"),
             "reclaim": ("bb_len", "bb_dev", "rsi_th"),
             "rsi_rev": ("rsi_th",),
             "donchian": ("dc_len",),
             "ema_pull": ("ema_f",),
             "momentum": ("mom_k", "mom_x")}


def describe(p):
    e = p.get("entry", "?")
    bits = [str(e)]
    for k in MODE_KEYS.get(e, ()):
        if k in p:
            bits.append(f"{k}={p[k]}")
    bits.append(f"filter={p.get('filter')}")
    bits.append(f"exit={p.get('exit')}")
    return " ".join(bits)


def signal_text(sig, p):
    d = "LONG 🟢" if sig["side"] > 0 else "SHORT 🔴"
    lines = [f"{sig['symbol']}  {d}  tf={p['tf']}",
             f"entry≈{sig['close']:.6g}  (bar {sig['bar']} UTC)"]
    if p["exit"] == "fixed":
        lines.append(f"TP≈{sig['tp']:.6g}  SL≈{sig['sl']:.6g}  "
                     f"time-stop {p['time_stop']} bars")
    else:
        lines.append(f"TP1≈{sig['tp1']:.6g} ({p['frac1']*100:.0f}% slice)  "
                     f"SL≈{sig['sl']:.6g}  breakeven after TP1")
        if "tp2" in sig:
            lines.append(f"runner TP2≈{sig['tp2']:.6g}  "
                         f"trail {p.get('trail_atr', 0)} ATR")
    lines.append(f"strategy: {describe(p)}")
    return "\n".join(lines)


# ------------------------------------------------------------------- news
def upcoming_news():
    """High/Medium USD events starting within the next 60 minutes."""
    req = urllib.request.Request(FF_CAL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        cal = json.loads(r.read())
    now = dt.datetime.now(dt.timezone.utc)
    out = []
    for e in cal:
        if e.get("country") != "USD":
            continue
        if e.get("impact") not in ("High", "Medium"):
            continue
        try:
            when = dt.datetime.fromisoformat(e["date"])
        except Exception:
            continue
        mins = (when - now).total_seconds() / 60.0
        if 0 <= mins <= NEWS_LOOKAHEAD_MIN:
            out.append((when, f"خبر USD تا {mins:.0f} دقیقه دیگر "
                              f"[{e.get('impact')}] {e['title']} "
                              f"(forecast {e.get('forecast') or '-'}, "
                              f"previous {e.get('previous') or '-'})"))
    return out


# ------------------------------------------------------------------- main
def run_once(p, st, last_news=0.0):
    sent = 0
    sigs = check_signals(p)
    for s in sigs:
        key = f"sig:{s['symbol']}:{p['tf']}:{s['bar']}:{s['side']}"
        if _seen(st, key):
            continue
        title = f"Signal {'LONG' if s['side'] > 0 else 'SHORT'} " \
                f"{s['symbol']} ({p['tf']})"
        text = signal_text(s, p)
        res = notify.send(title, text)
        notify.log_line("signal", text.replace("\n", " | "))
        sent += 1
        print(f"  SIGNAL {s['symbol']} {s['side']:+d} -> {res}")

    now = time.time()
    if now - last_news >= NEWS_POLL_SEC:
        last_news = now
        try:
            for when, text in upcoming_news():
                key = f"news:{when.isoformat()}:{text[:40]}"
                if _seen(st, key):
                    continue
                res = notify.send("News alert (USD)", text)
                notify.log_line("news", text)
                sent += 1
                print(f"  NEWS -> {res}")
        except Exception as e:
            print(f"  news calendar failed ({e})")
    return sent, last_news


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="single pass")
    ap.add_argument("--test", action="store_true", help="test notification")
    ap.add_argument("--config", default=None,
                    help="path to strategy JSON (default: results/"
                         "final_config.json; use results/best_config.json "
                         "for the v2 best strategy)")
    args = ap.parse_args()

    if args.test:
        res = notify.send("Test from trading-bot",
                          "این یک پیام آزمایشی است — اگر می‌بینید، کانال "
                          "اعلان فعال است.")
        print("channels:", json.dumps(res, ensure_ascii=False, indent=2))
        return

    cfg_path = args.config or CONFIG_PATH
    with open(cfg_path, encoding="utf-8") as f:
        p = json.load(f)["params"]
    print(f"strategy: tf={p['tf']} {describe(p)}")
    print(f"symbols: {len(bt.SYMBOLS)}  poll every {CHECK_SEC}s  "
          f"channels: {[k for k, v in notify.load_config().items() if v.get('enabled')]}")

    st = load_state()
    last_news = 0.0
    while True:
        t0 = time.time()
        n, last_news = run_once(p, st, last_news)
        save_state(st)
        now_utc = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
        print(f"[{now_utc.isoformat(timespec='seconds')}Z] "
              f"pass done: {n} notification(s), {time.time()-t0:.0f}s")
        if args.once:
            break
        time.sleep(CHECK_SEC)


if __name__ == "__main__":
    main()
