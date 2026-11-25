#!/usr/bin/env python3
"""FinMind 資料抓取（磁碟快取、可續跑、限速）。

用法：
  python fetch_finmind.py inst      # 三大法人買賣超（每檔一次呼叫，全歷史）
  python fetch_finmind.py revenue   # 月營收
  python fetch_finmind.py info      # TaiwanStockInfo（產業/市場別）

快取：backtest/data/finmind/<dataset>/<code>.csv.gz；已存在即跳過。
限速：預設每次請求間隔 14 秒（≈257 次/小時，低於免費 ~300 次/小時上限，
保留額度給實盤系統）。遇 402 / 額度訊息時睡 10 分鐘後重試。
"""
import gzip
import json
import os
import sys
import time
import urllib.error
import urllib.request

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "finmind")
API = "https://api.finmindtrade.com/api/v4/data"
PACE = float(os.environ.get("FINMIND_PACE", "14"))
START = "2014-10-01"

DATASETS = {
    "inst": "TaiwanStockInstitutionalInvestorsBuySell",
    "revenue": "TaiwanStockMonthRevenue",
    "info": "TaiwanStockInfo",
}


def universe():
    path = os.path.join(HERE, "universe.csv")
    return pd.read_csv(path, dtype={"code": str})["code"].tolist()


def _get(params):
    url = API + "?" + "&".join(f"{k}={v}" for k, v in params.items())
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 backtest"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))


def fetch(dataset, data_id=None, start=START):
    params = {"dataset": dataset}
    if data_id:
        params["data_id"] = data_id
    if start:
        params["start_date"] = start
    limit_hits = 0  # 402 退避：120s, 240s, 480s, 900s...
    for attempt in range(16):
        try:
            d = _get(params)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "ignore")[:200]
            print(f"  HTTP {e.code} {data_id} {time.strftime('%H:%M:%S')}: {body}", flush=True)
            if e.code in (402, 429):
                time.sleep(min(900, 120 * 2 ** limit_hits))
                limit_hits += 1
                continue
            time.sleep(60)
            continue
        except Exception as e:  # network hiccup
            print(f"  error {data_id}: {e}", flush=True)
            time.sleep(60)
            continue
        status = d.get("status")
        if status == 200:
            return d.get("data", [])
        msg = str(d.get("msg"))
        print(f"  status {status} {data_id}: {msg[:200]}", flush=True)
        if status == 402 or "limit" in msg.lower() or "upper" in msg.lower():
            time.sleep(min(900, 120 * 2 ** limit_hits))
            limit_hits += 1
        else:
            time.sleep(60)
    raise RuntimeError(f"failed {dataset} {data_id}")


def run(kind):
    dataset = DATASETS[kind]
    outdir = os.path.join(DATA, dataset)
    os.makedirs(outdir, exist_ok=True)
    if kind == "info":
        path = os.path.join(outdir, "all.csv.gz")
        if not os.path.exists(path):
            pd.DataFrame(fetch(dataset, start=None)).to_csv(path, index=False, compression="gzip")
        return
    codes = universe()
    todo = [c for c in codes if not os.path.exists(os.path.join(outdir, f"{c}.csv.gz"))]
    print(f"{dataset}: {len(codes)} codes, {len(todo)} to fetch", flush=True)
    for i, c in enumerate(todo):
        t0 = time.time()
        rows = fetch(dataset, c, START if kind == "inst" else "2013-01-01")
        tmp = os.path.join(outdir, f"{c}.tmp.gz")
        pd.DataFrame(rows).to_csv(tmp, index=False, compression="gzip")
        os.replace(tmp, os.path.join(outdir, f"{c}.csv.gz"))
        print(f"[{i+1}/{len(todo)}] {c} rows={len(rows)} {time.strftime('%H:%M:%S')}", flush=True)
        dt = time.time() - t0
        if i + 1 < len(todo):
            time.sleep(max(0.0, PACE - dt))


if __name__ == "__main__":
    for k in sys.argv[1:] or ["inst"]:
        run(k)
