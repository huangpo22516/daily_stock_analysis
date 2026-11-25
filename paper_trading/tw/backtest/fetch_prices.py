#!/usr/bin/env python3
"""Yahoo Finance 還原價（auto_adjust=True）抓取並快取。

輸出：backtest/data/prices/{open,close,volume}.csv.gz（寬表，欄=代碼，^TWII 也在內）
已存在即跳過；加 --force 重抓。
"""
import os
import sys

import pandas as pd
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "prices")
START = "2014-06-01"
END = "2026-11-26"


def main(force=False):
    os.makedirs(OUT, exist_ok=True)
    if not force and all(os.path.exists(os.path.join(OUT, f"{k}.csv.gz")) for k in ("open", "close", "volume")):
        print("prices cached")
        return
    codes = pd.read_csv(os.path.join(HERE, "universe.csv"), dtype={"code": str})["code"].tolist()
    tickers = [f"{c}.TW" for c in codes] + ["^TWII"]
    frames = {"Open": [], "Close": [], "Volume": []}
    for i in range(0, len(tickers), 50):
        batch = tickers[i:i + 50]
        df = yf.download(batch, start=START, end=END, auto_adjust=True, group_by="ticker",
                         threads=True, progress=False)
        for t in batch:
            if t not in df.columns.get_level_values(0):
                print("missing", t)
                continue
            sub = df[t]
            for k in frames:
                frames[k].append(sub[k].rename(t.replace(".TW", "")))
        print(f"batch {i // 50 + 1} done", flush=True)
    for k, lst in frames.items():
        wide = pd.concat(lst, axis=1).sort_index()
        wide.index = pd.to_datetime(wide.index).tz_localize(None).normalize()
        wide.to_csv(os.path.join(OUT, f"{k.lower()}.csv.gz"), compression="gzip")
        print(k, wide.shape)


if __name__ == "__main__":
    main("--force" in sys.argv)
