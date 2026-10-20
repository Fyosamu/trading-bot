"""Re-run stage4..stage7 grids on the current SYMBOLS list (shared store)."""
import backtest as bt
import stage4
import stage5
import stage6
import stage7

print(f"loading {len(bt.SYMBOLS)} symbols ...", flush=True)
store = {tf: {s: bt.load_symbol(s, tf) for s in bt.SYMBOLS}
         for tf in ("1h", "4h", "1d")}

stage4.stage4(store)   # trail exits -- high Win Rate zone
stage5.stage5(store)   # fixed TP=k*SL -- R:R ~ 2 zone
stage6.stage6(store)   # WR>=80 zone, best R:R inside it
stage7.stage7(store)   # R:R >= 1.5 zone, best Win Rate inside it
print("ALL GRIDS DONE", flush=True)
