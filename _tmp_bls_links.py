"""One-off: find prior-year BLS schedule page URLs + test live notify endpoints."""
import json
import re
import urllib.request

H = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

for page in ("cpi", "empsit"):
    url = f"https://www.bls.gov/schedule/news_release/{page}.htm"
    try:
        html = urllib.request.urlopen(
            urllib.request.Request(url, headers=H), timeout=20
        ).read().decode("utf-8", "ignore")
    except Exception as e:
        print(page, "FAIL", e)
        continue
    hrefs = sorted(set(re.findall(r'href="([^"]+)"', html)))
    print(page, "->", len(hrefs), "links; year/prior candidates:")
    for h in hrefs:
        if re.search(r"(20\d\d|prior)", h, re.I) and "css" not in h:
            print("   ", h)

# telegram bot API reachability (no token needed to see HTTP code)
try:
    r = urllib.request.urlopen(urllib.request.Request(
        "https://api.telegram.org/bot000:000/getMe", headers=H), timeout=10)
    print("telegram OK", r.status)
except Exception as e:
    print("telegram:", e)

# discord webhook host reachability
try:
    urllib.request.urlopen(urllib.request.Request(
        "https://discord.com/api/v10/gateway", headers=H), timeout=10)
    print("discord OK")
except Exception as e:
    print("discord:", e)

# tradingview host
try:
    urllib.request.urlopen(urllib.request.Request(
        "https://www.tradingview.com", headers=H), timeout=10)
    print("tradingview OK")
except Exception as e:
    print("tradingview:", e)
