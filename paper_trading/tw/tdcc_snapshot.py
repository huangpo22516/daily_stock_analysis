"""集保戶股權分散表週快照（opendata.tdcc.com.tw 只提供最新一週，所以每週存一份才能算變化）。

用法：python tdcc_snapshot.py   → 寫入 research/tdcc/<資料日期>.csv（已存在就略過）
欄位：code, big_pct（千張以上大戶持股比例 %）, small_pct（1 張以下 %）, holders（總人數）
dq_check.py 會比較最近兩週：大戶比例下降 ≥ 1 個百分點 且 總人數增加 ≥ 5% → 籌碼警訊（大戶出、散戶進）。
"""
import csv
import io
import os
import sys

import requests

URL = "https://opendata.tdcc.com.tw/getOD.ashx?id=1-5"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research", "tdcc")


def main():
    # 不帶 curl 式 User-Agent 會被導到 smart.tdcc.com.tw（不在白名單）
    txt = requests.get(URL, timeout=60, headers={"User-Agent": "curl/8.5.0"}, allow_redirects=False).content.decode("utf-8-sig")
    rows = list(csv.reader(io.StringIO(txt)))[1:]
    if not rows:
        sys.exit("TDCC 沒有資料")
    d = rows[0][0].strip()
    path = os.path.join(OUT, f"{d}.csv")
    if os.path.exists(path):
        print(f"已存在 {path}")
        return
    agg: dict[str, dict] = {}
    for r in rows:
        code, lvl = r[1].strip(), int(r[2])
        a = agg.setdefault(code, {})
        if lvl == 15:
            a["big_pct"] = float(r[5])
        elif lvl == 1:
            a["small_pct"] = float(r[5])
        elif lvl == 17:
            a["holders"] = int(r[3])
    os.makedirs(OUT, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["code", "big_pct", "small_pct", "holders"])
        for c, a in sorted(agg.items()):
            if "holders" in a:
                w.writerow([c, a.get("big_pct", ""), a.get("small_pct", ""), a["holders"]])
    print(f"寫入 {path}（{len(agg)} 檔，資料日期 {d}）")


if __name__ == "__main__":
    main()
