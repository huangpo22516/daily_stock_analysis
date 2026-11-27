"""資料品質與委託規則檢查（送出委託前、盤後研究後執行）。

用法：python dq_check.py [--date 下一交易日 YYYY-MM-DD]
檢查對象：待成交委託、持股、未結束的影子部位。
  1. 價格交叉比對：最新收盤 TWSE OpenAPI / FinMind / Yahoo 差異 > 0.5% 警告
  2. 漲跌幅合理性：單日漲跌超過 ±10.5% 視為資料錯誤
  3. 處置股 / 全額交割（變更交易）名單：買進委託命中 = ERROR
  4. 委託規則（PLAYBOOK 決策原則與風控）：升降單位、限價距收盤 ±10% 內、停損<限價<目標、
     報酬/風險 ≥ 1.5、目標報酬 ≥ 5 倍來回成本、單檔 ≤ 20%（ETF 40%）、單筆風險 ≤ 1.5%、
     必填 strategy/note/invalid/prob、事前機率推得的期望值（扣成本）> 0
  5. 籌碼乾淨度（免費替代指標，代替券商分點資料；2026-11-27 起）：
     a. 融資餘額 5 日增幅 > 20%（散戶追價）
     b. 當沖量占成交量 > 50%（隔日沖/短線主力）
     c. 量能 > 20 日均量 3 倍且單日漲跌 > 5%（爆量異動，需查有無消息）
     d. 集保千張大戶比例週減 ≥ 1 個百分點 且 總人數週增 ≥ 5%（大戶出、散戶進；需 tdcc_snapshot.py 累積兩週）
     買進委託命中 2 項以上 = ERROR；1 項 = WARN（報告裡要說明為何仍可買）
有 ERROR 時 exit code = 1。
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import date, timedelta

import requests

import engine as E

UA = {"User-Agent": "Mozilla/5.0"}
errors, warns = [], []


def err(msg):
    errors.append(msg)


def warn(msg):
    warns.append(msg)


def etf_tick(price: float) -> float:
    return 0.01 if price < 50 else 0.05


def valid_tick(code: str, price: float) -> bool:
    t = etf_tick(price) if E.is_etf(code) else E.tick(price)
    return abs(round(price / t) * t - price) < 1e-6


def latest_closes(code: str, before: str) -> dict:
    """回傳各來源最近一個交易日（< before）的 (date, close, change)。"""
    out = {}
    start = (date.fromisoformat(before) - timedelta(days=12)).isoformat()
    try:
        js = requests.get("https://api.finmindtrade.com/api/v4/data",
                          params=dict(dataset="TaiwanStockPrice", data_id=code, start_date=start), timeout=20).json()
        rows = [r for r in js.get("data", []) if r["date"] < before and r["Trading_Volume"] > 0]
        if rows:
            r = rows[-1]
            out["finmind"] = (r["date"], r["close"], r["spread"])
    except Exception as e:  # noqa: BLE001
        warn(f"{code} FinMind 讀取失敗：{e}")
    try:
        import yfinance as yf
        h = yf.Ticker(f"{code}.TW").history(start=start, auto_adjust=False)
        h = h[h.index.strftime("%Y-%m-%d") < before]
        if len(h):
            out["yahoo"] = (h.index[-1].strftime("%Y-%m-%d"), round(float(h["Close"].iloc[-1]), 2), None)
    except Exception as e:  # noqa: BLE001
        warn(f"{code} Yahoo 讀取失敗：{e}")
    for (c, d), r in E._twse_day_all(before).items():
        if c == code and d < before:
            try:
                out["twse"] = (d, float(r["ClosingPrice"]), float(r["Change"] or 0))
            except ValueError:
                pass
    return out


def restricted_codes(target: str) -> dict:
    """處置股（期間涵蓋 target）與變更交易/全額交割股。"""
    res = {}
    roc = lambda s: date(int(s[:3]) + 1911, int(s[4:6]), int(s[7:9])).isoformat()  # noqa: E731
    try:
        for r in requests.get("https://openapi.twse.com.tw/v1/announcement/punish", headers=UA, timeout=30).json():
            m = re.match(r"(\d+/\d+/\d+)～(\d+/\d+/\d+)", r.get("DispositionPeriod", ""))
            if m and roc(m.group(1)) <= target <= roc(m.group(2)):
                res[r["Code"].strip()] = f"處置股（{r['DispositionPeriod']}，{r['DispositionMeasures']}）"
    except Exception as e:  # noqa: BLE001
        warn(f"處置股名單讀取失敗：{e}")
    try:
        for r in requests.get("https://openapi.twse.com.tw/v1/exchangeReport/TWT85U", headers=UA, timeout=30).json():
            code = (r.get("Code") or r.get("證券代號") or "").strip()
            if code:
                res.setdefault(code, "變更交易/全額交割")
    except Exception as e:  # noqa: BLE001
        warn(f"變更交易名單讀取失敗：{e}")
    return res


FM = "https://api.finmindtrade.com/api/v4/data"


def _fm(ds, code, start, end):
    for _ in range(3):  # FinMind 免費額度偶爾回空或逾時，重試
        try:
            d = requests.get(FM, params=dict(dataset=ds, data_id=code, start_date=start, end_date=end),
                             timeout=30).json().get("data", [])
            if d:
                return d
        except Exception:
            pass
    return []


def chip_flags(code: str, before: str) -> list[str]:
    """籌碼乾淨度的免費替代指標（FinMind 免費資料）。抓不到資料就回傳空清單並另外警告。"""
    end = (date.fromisoformat(before) - timedelta(days=1)).isoformat()
    start = (date.fromisoformat(before) - timedelta(days=45)).isoformat()
    px = [r for r in _fm("TaiwanStockPrice", code, start, end) if r.get("Trading_Volume")]
    if len(px) < 21:
        warn(f"{code} 籌碼檢查：取不到足夠價量資料（FinMind），略過")
        return []
    flags = []
    last, prev = px[-1], px[-2]
    vol20 = sum(r["Trading_Volume"] for r in px[-21:-1]) / 20
    ret = last["close"] / prev["close"] - 1
    if last["Trading_Volume"] > 3 * vol20 and abs(ret) > 0.05:
        flags.append(f"爆量異動（量 {last['Trading_Volume'] / vol20:.1f} 倍、{ret:+.1%}）")
    # 當沖資料常晚一天更新（最新日為 0），取最近兩個交易日中有資料的一天
    dt = {r["date"]: r["Volume"] for r in _fm("TaiwanStockDayTrading", code, start, end) if r.get("Volume")}
    for r in (px[-1], px[-2]):
        if r["date"] in dt:
            if dt[r["date"]] / r["Trading_Volume"] > 0.5:
                flags.append(f"當沖比 {dt[r['date']] / r['Trading_Volume']:.0%}（{r['date']}）")
            break
    mg = _fm("TaiwanStockMarginPurchaseShortSale", code, start, end)
    if len(mg) >= 6 and mg[-6]["MarginPurchaseTodayBalance"] > 0:
        g = mg[-1]["MarginPurchaseTodayBalance"] / mg[-6]["MarginPurchaseTodayBalance"] - 1
        if g > 0.20:
            flags.append(f"融資 5 日 +{g:.0%}")
    t = _tdcc_change(code)
    if t:
        flags.append(t)
    return flags


def _tdcc_change(code: str) -> str | None:
    import csv
    import glob
    import os
    files = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "research", "tdcc", "*.csv")))
    if len(files) < 2:
        return None
    snap = []
    for f in files[-2:]:
        with open(f, encoding="utf-8") as fh:
            snap.append({r["code"]: r for r in csv.DictReader(fh)})
    a, b = snap[0].get(code), snap[1].get(code)
    if not (a and b and a["big_pct"] and b["big_pct"] and int(a["holders"]) > 0):
        return None
    dbig = float(b["big_pct"]) - float(a["big_pct"])
    dh = int(b["holders"]) / int(a["holders"]) - 1
    if dbig <= -1.0 and dh >= 0.05:
        return f"集保大戶比例 {dbig:+.1f}pt、人數 {dh:+.0%}"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="委託日（預設：待成交委託中最早的日期，否則明天）")
    a = ap.parse_args()
    led = E.load()
    pend = [o for o in led["orders"] if o["status"] == "PENDING"]
    target = a.date or (min(o["date"] for o in pend) if pend else (date.today() + timedelta(days=1)).isoformat())
    equity = led["equity"][-1]["equity"] if led["equity"] else led["initial_cash"]
    codes = {o["code"] for o in pend} | set(led["positions"]) | \
            {s["code"] for s in led.get("shadow", []) if s["status"] in ("WAIT", "OPEN")}
    restricted = restricted_codes(target)
    last_close = {}
    for c in sorted(codes):
        src = latest_closes(c, target)
        if not src:
            err(f"{c} 三個來源都拿不到收盤價")
            continue
        ds = {v[0] for v in src.values()}
        ref = src.get("twse") or src.get("finmind") or src.get("yahoo")
        last_close[c] = ref[1]
        if len(ds) > 1:
            warn(f"{c} 各來源最新交易日不一致：{ {k: v[0] for k, v in src.items()} }")
        same = [v[1] for v in src.values() if v[0] == ref[0]]
        if same and (max(same) / min(same) - 1) > 0.005:
            warn(f"{c} 收盤價來源差異 > 0.5%：{ {k: v[1] for k, v in src.items()} }")
        chg = ref[2]
        if chg is not None and ref[1] - chg > 0 and abs(chg / (ref[1] - chg)) > 0.105:
            err(f"{c} 單日漲跌 {chg / (ref[1] - chg) * 100:+.1f}% 超過漲跌停，資料可能有誤")
        if c in restricted:
            (err if any(o["code"] == c and o["side"] == "BUY" for o in pend) else warn)(f"{c} {restricted[c]}")
    # 籌碼乾淨度（買單與等待中的影子）
    buy_codes = {o["code"] for o in pend if o["side"] == "BUY"}
    for c in sorted(buy_codes | {s["code"] for s in led.get("shadow", []) if s["status"] == "WAIT"}):
        if E.is_etf(c):
            continue
        fl = chip_flags(c, target)
        if fl:
            msg = f"{c} 籌碼警訊：{'、'.join(fl)}"
            (err if (c in buy_codes and len(fl) >= 2) else warn)(msg)
    # 委託規則
    for o in pend:
        tag = f"#{o['id']} {o['side']} {o['code']}"
        if o["side"] != "BUY":
            continue
        for f in ("strategy", "note", "invalid"):
            if not o.get(f):
                err(f"{tag} 缺少 {f}")
        if o.get("prob") is None:
            warn(f"{tag} 缺少事前機率 prob")
        L, s, t = o.get("limit"), o.get("stop"), o.get("target")
        lc = last_close.get(o["code"])
        if L is None:
            warn(f"{tag} 市價單，無法檢查報酬/風險")
            continue
        if not valid_tick(o["code"], L):
            err(f"{tag} 限價 {L} 不符升降單位")
        if lc and not (lc * 0.9 <= L <= lc * 1.1):
            err(f"{tag} 限價 {L} 超出前收 {lc} 的 ±10%，不可能成交")
        if not (s and t and s < L < t):
            err(f"{tag} 需要 停損 < 限價 < 目標（{s} / {L} / {t}）")
            continue
        cost = E.roundtrip_cost(o["code"], L, o["shares"]) / (L * o["shares"])
        rr = (t - L) / (L - s)
        if rr < 1.5:
            err(f"{tag} 報酬/風險 {rr:.2f} < 1.5")
        if (t / L - 1) < 5 * cost:
            err(f"{tag} 目標報酬 {(t / L - 1) * 100:.1f}% < 5 倍來回成本 {cost * 500:.1f}%")
        pos = L * o["shares"] + (led["positions"].get(o["code"], {}).get("shares", 0) * L)
        cap = 0.40 if E.is_etf(o["code"]) else 0.20
        if pos > equity * cap:
            err(f"{tag} 部位 {pos:,.0f} 超過淨值 {cap:.0%}")
        risk = (L - s) * o["shares"]
        if risk > equity * 0.015:
            err(f"{tag} 單筆風險 {risk:,.0f} 超過淨值 1.5%")
        if o.get("prob") is not None:
            ev = o["prob"] * (t / L - 1) - (1 - o["prob"]) * (1 - s / L) - cost
            (warn if ev <= 0 else lambda m: None)(f"{tag} 以事前機率推得期望值 {ev * 100:+.2f}%（扣成本）≤ 0")
            o_ev = f"EV {ev * 100:+.2f}%"
        else:
            o_ev = ""
        print(f"OK  {tag} R/R {rr:.2f}｜目標 {(t / L - 1) * 100:+.1f}%｜停損 {(s / L - 1) * 100:+.1f}%｜成本 {cost * 100:.2f}%｜"
              f"部位 {pos / equity:.1%}｜風險 {risk / equity:.2%}｜{o_ev}")
    gross = sum(o["limit"] * o["shares"] for o in pend if o["side"] == "BUY" and o.get("limit")) + \
        sum(p.get("last_close", p["avg_cost"]) * p["shares"] for p in led["positions"].values())
    print(f"若買單全部成交，持股水位約 {gross / equity:.1%}")
    for w in warns:
        print("WARN", w)
    for e in errors:
        print("ERROR", e)
    print(f"檢查完成：{len(errors)} 個錯誤、{len(warns)} 個警告（委託日 {target}）")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
