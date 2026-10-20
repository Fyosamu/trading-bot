"""
Build data/news_events.csv -- high-impact US macro events (UTC timestamps).

Sources (all official, fetched and verified 2026-09):
  FOMC : federalreserve.gov/monetarypolicy/fomccalendars.htm
         statement released 14:00 ET on day 2 of each meeting
  CPI  : bls.gov/schedule/news_release/cpi.htm + /schedule/{2023,2024,2025}/home.htm
  NFP  : bls.gov/schedule/news_release/empsit.htm (Employment Situation)
         released 08:30 ET

Notes (kept honestly as-is from the official pages):
  * Oct 2025 CPI was never released (federal shutdown) -- not in BLS list.
  * Oct 2025 NFP never released; Sep-2025 NFP came late (Nov 20, 2025),
    Nov-2025 NFP came Dec 16, 2025.
  * Data ends mid-2026; events past the price-data end are dropped by the
    backtest automatically.
"""
import csv
import datetime as dt
import os
from zoneinfo import ZoneInfo

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "news_events.csv")

ET = ZoneInfo("America/New_York")

# (month, day) release dates, 08:30 ET
CPI_DATES = [
    # 2023
    (2023, 1, 12), (2023, 2, 14), (2023, 3, 14), (2023, 4, 12),
    (2023, 5, 10), (2023, 6, 13), (2023, 7, 12), (2023, 8, 10),
    (2023, 9, 13), (2023, 10, 12), (2023, 11, 14), (2023, 12, 12),
    # 2024
    (2024, 1, 11), (2024, 2, 13), (2024, 3, 12), (2024, 4, 10),
    (2024, 5, 15), (2024, 6, 12), (2024, 7, 11), (2024, 8, 14),
    (2024, 9, 11), (2024, 10, 10), (2024, 11, 13), (2024, 12, 11),
    # 2025 (Oct 2025 CPI skipped by BLS -- shutdown)
    (2025, 1, 15), (2025, 2, 12), (2025, 3, 12), (2025, 4, 10),
    (2025, 5, 13), (2025, 6, 11), (2025, 7, 15), (2025, 8, 12),
    (2025, 9, 11), (2025, 10, 24), (2025, 12, 18),
    # 2026
    (2026, 1, 13), (2026, 2, 13), (2026, 3, 11), (2026, 4, 10),
    (2026, 5, 12), (2026, 6, 10), (2026, 7, 14), (2026, 8, 12),
]

# Employment Situation (NFP), 08:30 ET
NFP_DATES = [
    # 2023
    (2023, 1, 6), (2023, 2, 3), (2023, 3, 10), (2023, 4, 7),
    (2023, 5, 5), (2023, 6, 2), (2023, 7, 7), (2023, 8, 4),
    (2023, 9, 1), (2023, 10, 6), (2023, 11, 3), (2023, 12, 8),
    # 2024
    (2024, 1, 5), (2024, 2, 2), (2024, 3, 8), (2024, 4, 5),
    (2024, 5, 3), (2024, 6, 7), (2024, 7, 5), (2024, 8, 2),
    (2024, 9, 6), (2024, 10, 4), (2024, 11, 1), (2024, 12, 6),
    # 2025 (Oct 2025 NFP never released; Sep-2025 delayed to Nov 20)
    (2025, 1, 10), (2025, 2, 7), (2025, 3, 7), (2025, 4, 4),
    (2025, 5, 2), (2025, 6, 6), (2025, 7, 3), (2025, 8, 1),
    (2025, 9, 5), (2025, 11, 20), (2025, 12, 16),
    # 2026
    (2026, 1, 9), (2026, 2, 11), (2026, 3, 6), (2026, 4, 3),
    (2026, 5, 8), (2026, 6, 5), (2026, 7, 2), (2026, 8, 7),
]

# FOMC statement dates (day 2 of meeting), 14:00 ET
FOMC_DATES = [
    # 2023
    (2023, 2, 1), (2023, 3, 22), (2023, 5, 3), (2023, 6, 14),
    (2023, 7, 26), (2023, 9, 20), (2023, 11, 1), (2023, 12, 13),
    # 2024
    (2024, 1, 31), (2024, 3, 20), (2024, 5, 1), (2024, 6, 12),
    (2024, 7, 25), (2024, 9, 18), (2024, 11, 7), (2024, 12, 18),
    # 2025
    (2025, 1, 29), (2025, 3, 19), (2025, 5, 7), (2025, 6, 18),
    (2025, 7, 30), (2025, 9, 17), (2025, 10, 29), (2025, 12, 10),
    # 2026
    (2026, 1, 28), (2026, 3, 18), (2026, 4, 29), (2026, 6, 17),
    (2026, 7, 29), (2026, 9, 16),
]


def build():
    rows = []
    for kind, dates, hour, minute in (
            ("CPI", CPI_DATES, 8, 30),
            ("NFP", NFP_DATES, 8, 30),
            ("FOMC", FOMC_DATES, 14, 0)):
        for y, m, d in dates:
            loc = dt.datetime(y, m, d, hour, minute, tzinfo=ET)
            utc = loc.astimezone(dt.timezone.utc)
            rows.append({
                "kind": kind,
                "name": f"{kind} {y}-{m:02d}-{d:02d}",
                "time_utc": utc.replace(tzinfo=None).isoformat(sep=" "),
            })
    rows.sort(key=lambda r: r["time_utc"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["kind", "name", "time_utc"])
        w.writeheader()
        w.writerows(rows)
    counts = {k: sum(1 for r in rows if r["kind"] == k)
              for k in ("FOMC", "CPI", "NFP")}
    print(f"wrote {OUT}: {len(rows)} events "
          f"(FOMC={counts['FOMC']} CPI={counts['CPI']} NFP={counts['NFP']})")
    print(f"first: {rows[0]['time_utc']}  last: {rows[-1]['time_utc']}")


if __name__ == "__main__":
    build()
