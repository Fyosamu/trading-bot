"""Unit test: hand-computed scenarios for the simulator."""
import numpy as np
import backtest as bt


def make_sd(op, hi, lo, cl):
    sd = {"time": np.array([np.datetime64("2024-01-01T00:00") + np.timedelta64(i, "h")
                            for i in range(len(cl))]),
          "open": np.array(op, float), "high": np.array(hi, float),
          "low": np.array(lo, float), "close": np.array(cl, float),
          "volume": np.ones(len(cl)), "_bb": {}}
    sd["rsi"] = np.full(len(cl), 50.0)
    sd["atr"] = np.full(len(cl), 1.0)     # ATR == 1 for easy math
    sd["ema200"] = np.full(len(cl), cl[0])
    return sd


def test_fixed_sl_full_loss():
    # entry at bar1 open=100, ATR=1, sl=2 -> SL at 98
    sd = make_sd([100, 100, 99, 98.5, 99],
                 [101, 101, 100, 99, 99.5],
                 [99.5, 99, 97.9, 98.4, 98.8],
                 [100, 99, 98.5, 99, 99.2])
    p = dict(exit="fixed", tp_atr=2.0, sl_atr=2.0, time_stop=50,
             tp1_atr=0, be_atr=0, tp2_atr=0, trail_atr=0, frac1=0.6)
    dirs = np.array([1, 0, 0, 0, 0], dtype=np.int8)
    tr = bt.simulate(dirs, sd, p)
    assert len(tr) == 1, tr
    exp = (98 - 100) / 100 - 2 * bt.COST
    assert abs(tr[0]["ret"] - exp) < 1e-9, (tr[0], exp)
    print("fixed SL full loss OK:", round(tr[0]["ret"], 6))


def test_scale_sl_before_partial():
    # full position must take the FULL loss (not the frac1 remainder)
    sd = make_sd([100, 100, 99, 97.9, 98],
                 [102, 100.6, 100, 98.5, 99],
                 [99.5, 99, 97.5, 97.8, 98.2],
                 [100, 99, 98, 98.2, 98.5])
    p = dict(exit="scale", sl_atr=2.0, time_stop=50,
             tp1_atr=1.0, be_atr=0.8, tp2_atr=3.0, trail_atr=0,
             frac1=0.6)
    dirs = np.array([1, 0, 0, 0, 0], dtype=np.int8)
    tr = bt.simulate(dirs, sd, p)
    exp = (98 - 100) / 100 - 2 * bt.COST
    assert tr[0]["reason"] == "SL", tr[0]
    assert abs(tr[0]["ret"] - exp) < 1e-9, (tr[0]["ret"], exp)
    print("scale SL-before-partial OK:", round(tr[0]["ret"], 6))


def test_scale_partial_then_be():
    # bar1: open 100, high 101 -> tp1 hit (101), partial done
    # bar2: low 99.5 -> BE exit for the remaining 40%
    sd = make_sd([100, 100, 100.5, 100.2],
                 [101, 101.4, 101, 100.8],
                 [99.5, 99.6, 99.9, 100.0],
                 [100.5, 100.6, 100.4, 100.5])
    p = dict(exit="scale", sl_atr=4.0, time_stop=50,
             tp1_atr=1.0, be_atr=0.8, tp2_atr=3.0, trail_atr=0,
             frac1=0.6)
    dirs = np.array([1, 0, 0, 0], dtype=np.int8)
    tr = bt.simulate(dirs, sd, p)
    leg1 = 0.6 * (101 - 100) / 100
    leg2 = 0.4 * (100 - 100) / 100
    exp = leg1 + leg2 - 2 * bt.COST
    assert tr[0]["reason"] == "BE", tr[0]
    assert abs(tr[0]["ret"] - exp) < 1e-9, (tr[0]["ret"], exp)
    print("scale partial+BE OK:", round(tr[0]["ret"], 6))


def test_trail_ratchets():
    # after partial, stop ratchets: high 104 -> trail 1.5 -> stop 102.5
    sd = make_sd([100, 100, 101, 102, 101.9],
                 [101, 104, 104.5, 105, 103],
                 [99.5, 100.5, 101.5, 102.4, 101.8],
                 [100.5, 103, 104, 103, 102.0])
    p = dict(exit="trail", sl_atr=3.0, time_stop=50,
             tp1_atr=1.0, be_atr=0.8, tp2_atr=0.0, trail_atr=1.5,
             frac1=0.5)
    dirs = np.array([1, 0, 0, 0, 0], dtype=np.int8)
    tr = bt.simulate(dirs, sd, p)
    # bar1 high=104 -> partial at 101; hh=104; stop = 104-1.5 = 102.5
    # bar2 low 101.5 <= 102.5 -> exit at 102.5 (TRAIL)
    assert tr[0]["reason"] == "TRAIL", tr[0]
    leg1 = 0.5 * (101 - 100) / 100
    leg2 = 0.5 * (102.5 - 100) / 100
    exp = leg1 + leg2 - 2 * bt.COST
    assert abs(tr[0]["ret"] - exp) < 1e-9, (tr[0]["ret"], exp)
    print("trail ratchet OK:", round(tr[0]["ret"], 6))


if __name__ == "__main__":
    test_fixed_sl_full_loss()
    test_scale_sl_before_partial()
    test_scale_partial_then_be()
    test_trail_ratchets()
    print("ALL TESTS PASSED")
