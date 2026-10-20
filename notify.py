"""
Multi-channel notification dispatch for bot signals.

Channels (all free / no trading-platform lock-in):
  ntfy     -- zero-account push: subscribe at https://ntfy.sh/<topic>
              from the phone app or browser (enabled by default)
  telegram -- BotFather token + your chat_id
  discord  -- incoming-webhook URL of a channel
  email    -- any SMTP account (Gmail app-password etc.)
  webhook  -- generic JSON POST (TradingView alert, Zapier, Make, IFTTT,
              3Commas/Cornix-style receivers)

Config file: notify_config.json (next to this file).  Only enabled
channels are called; a failing channel never blocks the others.
"""
import json
import os
import smtplib
import ssl
import urllib.request
from email.message import EmailMessage

HERE = os.path.dirname(os.path.abspath(__file__))
CFG_PATH = os.path.join(HERE, "notify_config.json")
LOG_PATH = os.path.join(HERE, "results", "signal_log.csv")

DEFAULT_CFG = {
    "ntfy": {"enabled": True, "topic": "trading-bot-sig-9k4d7x"},
    "telegram": {"enabled": False, "bot_token": "", "chat_id": ""},
    "discord": {"enabled": False, "webhook_url": ""},
    "email": {"enabled": False, "host": "smtp.gmail.com", "port": 587,
              "user": "", "password": "", "to": ""},
    "webhook": {"enabled": False, "url": ""},
}


def load_config():
    if not os.path.exists(CFG_PATH):
        return json.loads(json.dumps(DEFAULT_CFG))
    with open(CFG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    for k, v in DEFAULT_CFG.items():          # fill missing keys
        cfg.setdefault(k, json.loads(json.dumps(v)))
    return cfg


def _http(url, payload, headers=None, timeout=15):
    req = urllib.request.Request(
        url, data=payload,
        headers={"User-Agent": "trading-bot/1.0",
                 "Content-Type": "application/json; charset=utf-8",
                 **(headers or {})},
        method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status


def _ch_ntfy(c, title, msg):
    body = json.dumps({"topic": c["topic"], "title": title,
                       "message": msg, "priority": 4}).encode("utf-8")
    return _http("https://ntfy.sh", body)


def _ch_telegram(c, title, msg):
    text = f"*{_esc(title)}*\n{_esc(msg)}"
    body = json.dumps({"chat_id": c["chat_id"], "text": text,
                       "parse_mode": "MarkdownV2",
                       "disable_web_page_preview": True}).encode("utf-8")
    return _http(f"https://api.telegram.org/bot{c['bot_token']}/sendMessage",
                 body)


def _ch_discord(c, title, msg):
    body = json.dumps({"content": f"**{title}**\n{msg}"[:1900]}
                      ).encode("utf-8")
    return _http(c["webhook_url"], body)


def _ch_email(c, title, msg):
    m = EmailMessage()
    m["Subject"] = title
    m["From"] = c["user"]
    m["To"] = c["to"]
    m.set_content(msg)
    ctx = ssl.create_default_context()
    with smtplib.SMTP(c["host"], int(c["port"]), timeout=20) as s:
        s.starttls(context=ctx)
        s.login(c["user"], c["password"])
        s.send_message(m)
    return 250


def _ch_webhook(c, title, msg):
    body = json.dumps({"title": title, "message": msg,
                       "text": f"{title}\n{msg}"}).encode("utf-8")
    return _http(c["url"], body)


def _esc(t):  # minimal MarkdownV2 escaping for telegram
    for ch in r"_*[]()~`>#+-=|{}.!":
        t = t.replace(ch, "\\" + ch)
    return t


CHANNELS = {
    "ntfy": (_ch_ntfy, ("topic",)),
    "telegram": (_ch_telegram, ("bot_token", "chat_id")),
    "discord": (_ch_discord, ("webhook_url",)),
    "email": (_ch_email, ("host", "port", "user", "password", "to")),
    "webhook": (_ch_webhook, ("url",)),
}


def send(title, msg, cfg=None):
    """Dispatch to every enabled channel; returns {channel: status}."""
    cfg = cfg or load_config()
    out = {}
    for name, (fn, keys) in CHANNELS.items():
        c = cfg.get(name, {})
        if not c.get("enabled"):
            out[name] = "off"
            continue
        if any(not str(c.get(k, "")).strip() for k in keys):
            out[name] = "not-configured"
            continue
        try:
            code = fn(c, title, msg)
            out[name] = f"ok({code})"
        except Exception as e:
            out[name] = f"FAIL: {e}"
    log_line("notify", f"{title} | {msg[:400]} | {out}")
    return out


def log_line(kind, text):
    """Append one row to results/signal_log.csv (utf-8)."""
    import csv
    import datetime as dt
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    new = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time_utc", "kind", "text"])
        w.writerow([dt.datetime.now(dt.timezone.utc)
                    .replace(tzinfo=None).isoformat(timespec="seconds"),
                    kind, text])


if __name__ == "__main__":
    r = send("test signal", "این یک پیام آزمایشی از ربات ترید است.")
    print("channels:", r)
