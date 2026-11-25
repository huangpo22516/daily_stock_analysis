#!/usr/bin/env python3
"""Fetch official TWSE data (cached) for the screener.

Usage: python fetch_twse.py [YYYYMMDD]   (default: today)
- MI_INDEX (all listed daily quotes) for the last N_PRICE trading days
- T86 (institutional flows) for the last N_FLOW trading days
- notice / punish / TWT85U (attention, disposition, change-trading lists)
Trading-day calendar is taken from Yahoo ^TWII history (+ target date).
Raw files land in ./raw (one file per endpoint/date; re-runs only fetch missing).
"""
import json, os, sys, time, datetime as dt
import requests
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
os.makedirs(RAW, exist_ok=True)
N_PRICE = 22
N_FLOW = 10
SLEEP = 3.0
UA = {"User-Agent": "Mozilla/5.0 (screener)"}


def trading_days(target):
    h = yf.Ticker("^TWII").history(period="6mo", auto_adjust=False)
    days = sorted({d.strftime("%Y%m%d") for d in h.index})
    days = [d for d in days if d <= target]
    if target not in days:
        days.append(target)
    return days


def get(url, path, validate):
    if os.path.exists(path):
        try:
            if validate(json.load(open(path))):
                return True
        except Exception:
            pass
    for attempt in range(3):
        try:
            r = requests.get(url, headers=UA, timeout=60)
            d = r.json()
            if validate(d):
                json.dump(d, open(path, "w"), ensure_ascii=False)
                time.sleep(SLEEP)
                return True
            print("  invalid payload", url, str(d)[:120])
        except Exception as e:
            print("  error", url, e)
        time.sleep(SLEEP * (attempt + 2))
    return False


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else dt.date.today().strftime("%Y%m%d")
    days = trading_days(target)
    json.dump(days, open(os.path.join(RAW, "calendar.json"), "w"))
    ok = lambda d: isinstance(d, dict) and d.get("stat") == "OK"
    for d in days[-N_PRICE:]:
        print("MI_INDEX", d)
        get(f"https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX?date={d}&type=ALLBUT0999&response=json",
            os.path.join(RAW, f"mi_{d}.json"), ok)
    for d in days[-N_FLOW:]:
        print("T86", d)
        get(f"https://www.twse.com.tw/rwd/zh/fund/T86?date={d}&selectType=ALLBUT0999&response=json",
            os.path.join(RAW, f"t86_{d}.json"), ok)
    # status lists (always refresh: they are "as of now")
    for name, url in [
        ("notice", "https://www.twse.com.tw/rwd/zh/announcement/notice?response=json"),
        ("punish", "https://www.twse.com.tw/rwd/zh/announcement/punish?response=json"),
    ]:
        p = os.path.join(RAW, f"{name}_{target}.json")
        if os.path.exists(p):
            os.remove(p)
        get(url, p, ok)
    for name, path in [("punish_oa", "announcement/punish"), ("twt85u", "exchangeReport/TWT85U"),
                       ("notetrans", "announcement/notetrans")]:
        try:
            r = requests.get(f"https://openapi.twse.com.tw/v1/{path}", headers=UA, timeout=60)
            json.dump(r.json(), open(os.path.join(RAW, f"{name}_{target}.json"), "w"), ensure_ascii=False)
        except Exception as e:
            print("openapi error", path, e)
        time.sleep(1)
    print("done; days:", days[-N_PRICE:])


if __name__ == "__main__":
    main()
