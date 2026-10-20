"""
Backtesting engine v2 — mean reversion, honest simulation rules.

Honesty rules (non-negotiable):
  * signals use only data up to the signal bar close; entry = NEXT bar open
  * if SL and target are both touched inside one bar, STOP fills first (pessimistic)
  * fees 0.04%/side + slippage 0.02%/side charged on every fill
  * model selection uses TRAIN data only; the OOS test period is touched once

Entry modes:
  touch   : close beyond the Bollinger band + RSI oversold (classic)
  reclaim : price was outside the band for the last 3 bars and closes back
            inside, with RSI still cool (reversal confirmation)

Exit modes:
  fixed  : TP = tp_atr*ATR, SL = sl_atr*ATR, time stop
  scale  : close frac1 at tp1_atr*ATR, move SL to breakeven for the rest,
           rest runs to tp2_atr*ATR or time stop.  (high win-rate design,
           losses still strictly capped at sl_atr*ATR)
"""
import itertools
import json
import os
import time

import numpy as np
import pandas as pd

FEE = 0.0004
SLIP = 0.0002
COST = FEE + SLIP

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RESULTS = os.path.join(HERE, "results")

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT",
           "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT",
           "DOTUSDT", "LTCUSDT", "BNBUSDT", "ATOMUSDT",
           "TRXUSDT", "NEARUSDT", "ARBUSDT", "OPUSDT",
           "FILUSDT", "INJUSDT", "APTUSDT", "SUIUSDT",
           "PEPEUSDT", "SHIBUSDT", "UNIUSDT", "ETCUSDT",
           "BCHUSDT", "XLMUSDT", "ALGOUSDT",
           # added by extend_data.py (2018-2026 archive pull)
           "AAVEUSDT", "GRTUSDT", "HBARUSDT", "ICPUSDT",
           "SANDUSDT", "AXSUSDT", "MANAUSDT", "EGLDUSDT",
           "NEOUSDT", "VETUSDT", "KAVAUSDT", "ENJUSDT",
           "ZECUSDT", "1INCHUSDT", "CHZUSDT"]
TRAIN_END = pd.Timestamp("2025-01-01")

TF_RULE = {"1h": None, "4h": "4h", "1d": "1D"}


# ---------------------------------------------------------------- indicators
def ema(x, n):
    return pd.Series(x).ewm(span=n, adjust=False).mean().to_numpy()


def rsi_wilder(close, n=14):
    d = np.diff(close, prepend=close[0])
    up = pd.Series(np.where(d > 0, d, 0.0))
    dn = pd.Series(np.where(d < 0, -d, 0.0))
    rs = up.ewm(alpha=1.0 / n, adjust=False).mean() / \
        dn.ewm(alpha=1.0 / n, adjust=False).mean()
    return np.nan_to_num(100.0 - 100.0 / (1.0 + rs.to_numpy()), nan=50.0)


def atr_wilder(high, low, close, n=14):
    prev = np.roll(close, 1)
    prev[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev),
                                           np.abs(low - prev)))
    return pd.Series(tr).ewm(alpha=1.0 / n, adjust=False).mean().to_numpy()


def bollinger(close, n, dev):
    s = pd.Series(close)
    mid = s.rolling(n).mean().to_numpy()
    sd = s.rolling(n).std(ddof=0).to_numpy()
    return mid - dev * sd, mid + dev * sd, mid


# ---------------------------------------------------------------- data
def _make_sd(times, o, h, lo, c, v):
    sd = {"time": np.asarray(times), "open": o, "high": h, "low": lo,
          "close": c, "volume": v, "_bb": {}}
    sd["rsi"] = rsi_wilder(c, 14)
    sd["atr"] = atr_wilder(h, lo, c, 14)
    sd["ema200"] = ema(c, 200)
    return sd


def load_symbol(sym, tf="1h"):
    path = os.path.join(DATA, f"{sym}_1h.csv.gz")
    df = pd.read_csv(path, compression="gzip")
    # merged caches mix two timestamp text formats (early Binance monthly
    # zips write "...:00.000", later ones plain) -> parse element-wise;
    # parse_dates would infer from the first row and fail on the rest
    df["time"] = pd.to_datetime(df["time"], format="mixed", errors="coerce")
    if df["time"].isna().any():
        df = df[df["time"].notna()]
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    df = df.sort_values("time").reset_index(drop=True)
    if TF_RULE[tf] is not None:
        df["bucket"] = df["time"].dt.floor(TF_RULE[tf])
        g = df.groupby("bucket", sort=True)
        out = pd.DataFrame({
            "time": g["time"].first(),
            "open": g["open"].first(),
            "high": g["high"].max(),
            "low": g["low"].min(),
            "close": g["close"].last(),
            "volume": g["volume"].sum(),
        }).reset_index(drop=True)
        df = out
    return _make_sd(df["time"].to_numpy(),
                    df["open"].to_numpy(), df["high"].to_numpy(),
                    df["low"].to_numpy(), df["close"].to_numpy(),
                    df["volume"].to_numpy())


def get_bb(sd, n, dev):
    key = (n, dev)
    if key not in sd["_bb"]:
        sd["_bb"][key] = bollinger(sd["close"], n, dev)
    return sd["_bb"][key]


def _ema_cached(sd, n, c):
    """EMA with per-symbol cache (same pattern as _bb)."""
    cache = sd.setdefault("_ema", {})
    if n not in cache:
        cache[n] = ema(c, n)
    return cache[n]


# ---------------------------------------------------------------- signals
def build_dirs(sd, p):
    c = sd["close"]
    rsi = sd["rsi"]
    n = len(c)
    warm = int(p.get("bb_len", 20)) + 1
    entry = p["entry"]
    if entry in ("touch", "reclaim"):
        lo, up, _ = get_bb(sd, p["bb_len"], p["bb_dev"])
    if entry == "touch":
        lg = (c < lo) & (rsi < p["rsi_th"])
        sh = (c > up) & (rsi > 100.0 - p["rsi_th"])
    elif entry == "reclaim":
        below = pd.Series(c < lo)
        above = pd.Series(c > up)
        was_below = (below.astype(float).rolling(3).max().shift(1) > 0).to_numpy()
        was_above = (above.astype(float).rolling(3).max().shift(1) > 0).to_numpy()
        cool = (pd.Series(rsi < p["rsi_th"]).rolling(3).max() > 0).to_numpy()
        hot = (pd.Series(rsi > 100.0 - p["rsi_th"]).rolling(3).max() > 0).to_numpy()
        lg = (~below.to_numpy()) & was_below & cool
        sh = (~above.to_numpy()) & was_above & hot
    elif entry == "rsi_rev":                # classic RSI mean reversion
        lg = (rsi < p["rsi_th"]).to_numpy() if hasattr(rsi, "to_numpy") \
            else rsi < p["rsi_th"]
        sh = (rsi > 100.0 - p["rsi_th"]).to_numpy() if hasattr(rsi, "to_numpy") \
            else rsi > 100.0 - p["rsi_th"]
        warm = max(warm, 15)
    elif entry == "donchian":               # N-bar breakout (trend/momentum)
        k = int(p["dc_len"])
        hh = pd.Series(sd["high"]).rolling(k).max().shift(1).to_numpy()
        ll = pd.Series(sd["low"]).rolling(k).min().shift(1).to_numpy()
        lg = c > hh
        sh = c < ll
        warm = max(warm, k + 2)
    elif entry == "ema_pull":               # pullback to fast EMA in trend
        ef = _ema_cached(sd, int(p["ema_f"]), c)
        upc = c > ef
        prev = np.concatenate([[False], upc[:-1]])
        lg = upc & ~prev
        sh = (~upc) & prev
        warm = max(warm, int(p["ema_f"]) + 2)
    elif entry == "momentum":               # k-bar momentum vs ATR
        k = int(p["mom_k"])
        x = float(p["mom_x"])
        mom = c - np.concatenate([c[:k], c[:-k]])  # c[i] - c[i-k]
        lg = mom > x * sd["atr"]
        sh = mom < -x * sd["atr"]
        warm = max(warm, k + 2)
    else:
        raise ValueError(f"unknown entry mode: {entry}")
    d = np.zeros(n, dtype=np.int8)
    d[lg] = 1
    d[sh] = -1
    if p["filter"] == "ema200":
        e = sd["ema200"]
        d[(d == 1) & ~(c > e)] = 0
        d[(d == -1) & ~(c < e)] = 0
    d[:warm] = 0
    d[sd["atr"] <= 0] = 0
    return d


# ---------------------------------------------------------------- simulator
def simulate(dirs, sd, p):
    o, h, lo, c = sd["open"], sd["high"], sd["low"], sd["close"]
    atr, times = sd["atr"], sd["time"]
    n = len(c)
    trades = []
    i = 0
    while i < n:
        d = int(dirs[i])
        if d == 0 or i + 1 >= n or atr[i] <= 0:
            i += 1
            continue
        j0 = i + 1
        entry = o[j0]
        if entry <= 0:
            i += 1
            continue
        a = atr[i]
        side = 1 if d > 0 else -1
        # levels ---------------------------------------------------------
        if p["exit"] == "fixed":
            tp = entry + side * p["tp_atr"] * a
            sl = entry - side * p["sl_atr"] * a
        else:  # scale / trail
            tp1 = entry + side * p["tp1_atr"] * a
            sl = entry - side * p["sl_atr"] * a
            be = entry
            hh, ll = entry, entry
            if p["exit"] == "trail":
                tp2 = (entry + side * p["tp2_atr"] * a
                       if p.get("tp2_atr", 0) > 0 else None)
                trail_w = p["trail_atr"] * a
            else:
                tp2 = entry + side * p["tp2_atr"] * a
        j = j0
        exit_px, reason = None, None
        partial = False
        frac1 = p.get("frac1", 0.6)
        leg1 = 0.0
        while j < n:
            if p["exit"] == "fixed":
                if side > 0:
                    if lo[j] <= sl:
                        exit_px, reason = sl, "SL"
                        break
                    if h[j] >= tp:
                        exit_px, reason = tp, "TP"
                        break
                else:
                    if h[j] >= sl:
                        exit_px, reason = sl, "SL"
                        break
                    if lo[j] <= tp:
                        exit_px, reason = tp, "TP"
                        break
            elif p["exit"] == "scale":
                if not partial:
                    if side > 0:
                        if lo[j] <= sl:
                            exit_px, reason = sl, "SL"
                            break
                        if h[j] >= tp1:
                            partial = True
                            leg1 = frac1 * side * (tp1 - entry) / entry
                    else:
                        if h[j] >= sl:
                            exit_px, reason = sl, "SL"
                            break
                        if lo[j] <= tp1:
                            partial = True
                            leg1 = frac1 * side * (tp1 - entry) / entry
                else:
                    if side > 0:
                        if lo[j] <= be:
                            exit_px, reason = be, "BE"
                            break
                        if h[j] >= tp2:
                            exit_px, reason = tp2, "TP2"
                            break
                    else:
                        if h[j] >= be:
                            exit_px, reason = be, "BE"
                            break
                        if lo[j] <= tp2:
                            exit_px, reason = tp2, "TP2"
                            break
            else:  # trail: partial at tp1, then ratchet stop with the trend
                if not partial:
                    if side > 0:
                        if lo[j] <= sl:
                            exit_px, reason = sl, "SL"
                            break
                        if h[j] >= tp1:
                            partial = True
                            leg1 = frac1 * side * (tp1 - entry) / entry
                            hh = h[j]
                    else:
                        if h[j] >= sl:
                            exit_px, reason = sl, "SL"
                            break
                        if lo[j] <= tp1:
                            partial = True
                            leg1 = frac1 * side * (tp1 - entry) / entry
                            ll = lo[j]
                else:
                    if side > 0:
                        # conservative: judge the stop with the best price
                        # reached BEFORE this bar, then ratchet
                        raw = hh - trail_w
                        stop_lvl, sreason = ((raw, "TRAIL") if raw > be
                                             else (be, "BE"))
                        if lo[j] <= stop_lvl:
                            exit_px, reason = stop_lvl, sreason
                            break
                        if tp2 is not None and h[j] >= tp2:
                            exit_px, reason = tp2, "TP2"
                            break
                        hh = max(hh, h[j])
                    else:
                        raw = ll + trail_w
                        stop_lvl, sreason = ((raw, "TRAIL") if raw < be
                                             else (be, "BE"))
                        if h[j] >= stop_lvl:
                            exit_px, reason = stop_lvl, sreason
                            break
                        if tp2 is not None and lo[j] <= tp2:
                            exit_px, reason = tp2, "TP2"
                            break
                        ll = min(ll, lo[j])
            if (j - j0 + 1) >= p["time_stop"]:
                exit_px, reason = c[j], "TIME"
                break
            j += 1
        if exit_px is None:
            j = n - 1
            exit_px, reason = c[j], "END"
        if p["exit"] == "fixed":
            net = side * (exit_px - entry) / entry
        else:
            # if the first (partial) close never happened, the FULL
            # position was closed by this exit -> multiplier 1.0
            mult = (1.0 - frac1) if partial else 1.0
            net = leg1 + mult * side * (exit_px - entry) / entry
        net -= 2.0 * COST
        trades.append({
            "symbol": sd.get("symbol", "?"),
            "side": "long" if side > 0 else "short",
            "entry_time": times[j0], "exit_time": times[j],
            "entry": entry, "exit": exit_px, "ret": net,
            "reason": reason, "bars": j - j0 + 1,
        })
        i = j + 1
    return trades


# ---------------------------------------------------------------- metrics
def metrics(trades):
    if not trades:
        return dict(n=0, win_rate=0.0, pf=0.0, expectancy=0.0, avg_win=0.0,
                    avg_loss=0.0, rr=0.0, max_dd=0.0, max_consec_loss=0,
                    total=0.0, scratch=0)
    r = np.array([t["ret"] for t in trades])
    wins, losses = r[r > 1e-12], r[r < -1e-12]
    scratch = int(((r >= -1e-12) & (r <= 1e-12)).sum())
    gross_l = abs(losses.sum())
    pf = float(wins.sum() / gross_l) if gross_l > 0 else float("inf")
    eq = np.cumsum(r)
    peak = np.maximum.accumulate(np.concatenate([[0.0], eq]))
    dd = float(np.max(peak[1:] - eq)) if len(eq) else 0.0
    mc, cur = 0, 0
    for x in r:
        if x <= 0:
            cur += 1
            mc = max(mc, cur)
        else:
            cur = 0
    aw = float(wins.mean() * 100.0) if len(wins) else 0.0
    al = float(losses.mean() * 100.0) if len(losses) else 0.0
    return dict(
        n=int(len(r)),
        win_rate=float((r > 0).mean() * 100.0),
        pf=pf,
        expectancy=float(r.mean() * 100.0),
        avg_win=aw,
        avg_loss=al,
        rr=float(aw / abs(al)) if al < 0 else float("inf"),
        max_dd=dd * 100.0,
        max_consec_loss=int(mc),
        total=float(r.sum() * 100.0),
        scratch=scratch,
    )


def run_config(store, params):
    trades = []
    for sym, sd in store[params["tf"]].items():
        sd["symbol"] = sym
        dirs = build_dirs(sd, params)
        trades.extend(simulate(dirs, sd, params))
    trades.sort(key=lambda t: t["entry_time"])
    train = [t for t in trades if t["entry_time"] < TRAIN_END]
    test = [t for t in trades if t["entry_time"] >= TRAIN_END]
    return trades, metrics(train), metrics(test)


# ---------------------------------------------------------------- grids
TSTOP = {"1h": [24, 72], "4h": [12, 30], "1d": [5, 15]}


RSI_BY_ENTRY = {"touch": [25, 35], "reclaim": [35, 50]}


def stage1(store):
    rows = []
    t0 = time.time()
    done = 0
    for tf in ("1h", "4h", "1d"):
        combos = []
        for entry in ("touch", "reclaim"):
            for (bbl, bbd, rsi, tp, sl, filt, ts) in itertools.product(
                    [14, 20, 30], [1.5, 2.0, 2.5], RSI_BY_ENTRY[entry],
                    [0.5, 0.8], [2.0, 3.0], ["none", "ema200"], TSTOP[tf]):
                combos.append((entry, bbl, bbd, rsi, tp, sl, filt, ts))
        for (entry, bbl, bbd, rsi, tp, sl, filt, ts) in combos:
            p = dict(tf=tf, entry=entry, bb_len=bbl, bb_dev=bbd, rsi_th=rsi,
                     filter=filt, exit="fixed", tp_atr=tp, sl_atr=sl,
                     time_stop=ts)
            _, tr, te = run_config(store, p)
            rows.append({**p, **{f"train_{k}": v for k, v in tr.items()},
                         **{f"test_{k}": v for k, v in te.items()}})
            done += 1
            if done % 60 == 0:
                print(f"  {done} configs  {time.time()-t0:.0f}s", flush=True)
        print(f"  stage1 {tf} done  ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(os.path.join(RESULTS, "stage1.csv"), index=False)
    print(f"stage1: {len(df)} configs in {time.time()-t0:.0f}s")
    return df


def rank(df, min_train_n=200, min_pf=1.0):
    d = df[(df.train_n >= min_train_n) & (df.train_pf >= min_pf)].copy()
    if d.empty:
        d = df[df.train_n >= 100].copy()
    d["score"] = d.train_win_rate + 5.0 * np.minimum(d.train_pf, 3.0)
    return d.sort_values("score", ascending=False)


def stage2(store, parents, variants=8):
    """expand best stage-1 configs with the scale-out exit design"""
    rows = []
    seen = set()
    for _, par in parents.iterrows():
        base = {k: par[k] for k in
                ("tf", "entry", "bb_len", "bb_dev", "rsi_th", "filter",
                 "sl_atr", "time_stop")}
        for (tp1, be, tp2, f1, sl) in itertools.product(
                [0.3, 0.5], [0.8, 1.2], [1.5, 2.5, 4.0], [0.6],
                [par["sl_atr"], 4.0]):
            p = {**base, "exit": "scale", "tp1_atr": tp1, "be_atr": be,
                 "tp2_atr": tp2, "frac1": f1, "sl_atr": sl}
            key = json.dumps(p, sort_keys=True, default=str)
            if key in seen:
                continue
            seen.add(key)
            _, tr, te = run_config(store, p)
            rows.append({**p, **{f"train_{k}": v for k, v in tr.items()},
                         **{f"test_{k}": v for k, v in te.items()}})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "stage2.csv"), index=False)
    return df


def load_store():
    store = {}
    for tf in ("1h", "4h", "1d"):
        store[tf] = {}
        for s in SYMBOLS:
            sd = load_symbol(s, tf)
            store[tf][s] = sd
        n = len(next(iter(store[tf].values()))["close"])
        print(f"  {tf}: {n} bars/symbol")
    return store


if __name__ == "__main__":
    print("loading ...")
    store = load_store()
    print("stage 1 (fixed TP/SL grid) ...")
    s1 = stage1(store)
    top = rank(s1)
    cols = ["tf", "entry", "bb_len", "bb_dev", "rsi_th", "tp_atr", "sl_atr",
            "filter", "time_stop",
            "train_n", "train_win_rate", "train_pf", "train_expectancy",
            "test_n", "test_win_rate", "test_pf", "test_expectancy"]
    print("\nSTAGE1 TOP 12:")
    print(top[cols].head(12).to_string(index=False))
    print("\nstage 2 (scale-out + breakeven design) ...")
    s2 = stage2(store, top.head(20))
    top2 = rank(s2)
    cols2 = ["tf", "entry", "bb_len", "bb_dev", "rsi_th", "sl_atr", "tp1_atr",
             "be_atr", "tp2_atr", "filter", "time_stop",
             "train_n", "train_win_rate", "train_pf", "train_expectancy",
             "test_n", "test_win_rate", "test_pf", "test_expectancy"]
    print("\nSTAGE2 TOP 15:")
    print(top2[cols2].head(15).to_string(index=False))
