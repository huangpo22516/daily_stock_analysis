"""台股模擬盤引擎（現股，上市股票/ETF，可零股、可現股當沖）。

用法：
  python engine.py init [--cash 100000] [--start 2026-11-26]
  python engine.py order BUY 2330 10 --limit 1000 [--stop 920] [--target 1150] [--note "..."] [--date 2026-11-26]
  python engine.py order SELL 2330 10 [--limit 1100] [--date ...]      # 不給 --limit = 開盤市價
  python engine.py cancel <order_id>
  python engine.py settle [--date 2026-11-26]   # 用當日 OHLC 撮合委託、檢查停損/停利、記錄淨值
  python engine.py status

撮合規則（保守近似）：
  - 買進限價 L：開盤 <= L 以開盤價成交；否則當日最低 <= L 以 L 成交；否則未成交（當日有效）。
  - 賣出限價 L：開盤 >= L 以開盤價成交；否則當日最高 >= L 以 L 成交。
  - 市價單：以開盤價成交，加滑價 SLIPPAGE。
  - 持股停損 S（收盤後檢查當日盤中）：開盤 <= S 以開盤價出場；否則最低 <= S 以 S 減滑價出場。
    同日同時觸及停損與停利時，假設先觸停損（保守）。
  - 零股以整股日 K 近似（盤中零股實際價差會略差，另加滑價）。
費用：手續費 0.1425%（整股最低 20 元、零股最低 1 元）；證交稅 股票 0.3%、當沖 0.15%、ETF 0.1%。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import date, datetime
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
LEDGER = HERE / "ledger.json"

FEE_RATE = 0.001425
MIN_FEE_BOARD = 20
MIN_FEE_ODD = 1
TAX_STOCK = 0.003
TAX_DAYTRADE = 0.0015
TAX_ETF = 0.001
SLIPPAGE = 0.002  # 市價/停損/零股的滑價假設

UA = {"User-Agent": "Mozilla/5.0"}


# ---------------------------------------------------------------- utilities
def load() -> dict:
    return json.loads(LEDGER.read_text(encoding="utf-8"))


def save(led: dict) -> None:
    LEDGER.write_text(json.dumps(led, ensure_ascii=False, indent=2), encoding="utf-8")


def is_etf(code: str) -> bool:
    return code.startswith("00")


def fee(amount: float, shares: int) -> int:
    minimum = MIN_FEE_BOARD if shares % 1000 == 0 else MIN_FEE_ODD
    return max(minimum, math.floor(amount * FEE_RATE))


def tax(amount: float, code: str, daytrade: bool) -> int:
    rate = TAX_ETF if is_etf(code) else (TAX_DAYTRADE if daytrade else TAX_STOCK)
    return math.floor(amount * rate)


def tick(price: float) -> float:
    """台股升降單位（股票）。ETF 規則不同，這裡只用來把滑價後價格取整。"""
    for lim, t in ((10, 0.01), (50, 0.05), (100, 0.1), (500, 0.5), (1000, 1.0)):
        if price < lim:
            return t
    return 5.0


def round_tick(price: float, up: bool) -> float:
    t = tick(price)
    n = price / t
    return round((math.ceil(n - 1e-9) if up else math.floor(n + 1e-9)) * t, 2)


_ohlc_cache: dict[tuple[str, str], dict | None] = {}


_day_all: dict | None = None


def _twse_day_all(d: str) -> dict:
    """TWSE OpenAPI 全市場當日行情（只有最新交易日；晚間才更新）。"""
    global _day_all
    if _day_all is None:
        _day_all = {}
        try:
            rows = requests.get("https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL",
                                headers=UA, timeout=40).json()
            for r in rows:
                roc = r["Date"]
                iso = f"{int(roc[:3]) + 1911}-{roc[3:5]}-{roc[5:]}"
                _day_all[(r["Code"], iso)] = r
        except Exception as e:  # noqa: BLE001
            print(f"[warn] TWSE STOCK_DAY_ALL: {e}", file=sys.stderr)
    return _day_all


def ohlc(code: str, d: str) -> dict | None:
    """回傳 {'open','high','low','close','volume','change'}。
    來源順序：TWSE OpenAPI(最新日) → FinMind → Yahoo。"""
    key = (code, d)
    if key in _ohlc_cache:
        return _ohlc_cache[key]
    res = None
    r = _twse_day_all(d).get((code, d))
    if r and r.get("OpeningPrice") not in ("", "--", None):
        try:
            res = dict(open=float(r["OpeningPrice"]), high=float(r["HighestPrice"]),
                       low=float(r["LowestPrice"]), close=float(r["ClosingPrice"]),
                       volume=float(r["TradeVolume"]), change=float(r["Change"] or 0))
        except ValueError:
            res = None
    if res is None:
        try:
            js = requests.get("https://api.finmindtrade.com/api/v4/data",
                              params=dict(dataset="TaiwanStockPrice", data_id=code, start_date=d, end_date=d),
                              timeout=20).json()
            for row in js.get("data", []):
                if row["date"] == d and row["Trading_Volume"] > 0:
                    res = dict(open=row["open"], high=row["max"], low=row["min"], close=row["close"],
                               volume=row["Trading_Volume"], change=row["spread"])
        except Exception as e:  # noqa: BLE001
            print(f"[warn] FinMind {code} {d}: {e}", file=sys.stderr)
    if res is None:
        try:
            import yfinance as yf
            h = yf.Ticker(f"{code}.TW").history(start=d, period="5d")
            h = h[h.index.strftime("%Y-%m-%d") == d]
            if len(h) and h["Volume"].iloc[0] > 0:
                r = h.iloc[0]
                res = dict(open=float(r.Open), high=float(r.High), low=float(r.Low),
                           close=float(r.Close), volume=float(r.Volume), change=None)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] yahoo {code} {d}: {e}", file=sys.stderr)
    _ohlc_cache[key] = res
    return res


def locked(k: dict) -> str | None:
    """一價到底的漲停/跌停鎖死日：買單（漲停）或賣單（跌停）視為排不到。"""
    if k["open"] == k["high"] == k["low"] == k["close"] and k.get("change"):
        prev = k["close"] - k["change"]
        if prev > 0 and k["change"] / prev >= 0.09:
            return "UP"
        if prev > 0 and k["change"] / prev <= -0.09:
            return "DOWN"
    return None


def index_close(d: str) -> float | None:
    try:
        import yfinance as yf
        h = yf.Ticker("^TWII").history(start=d, period="5d")
        h = h[h.index.strftime("%Y-%m-%d") == d]
        if len(h):
            return round(float(h["Close"].iloc[0]), 2)
    except Exception:  # noqa: BLE001
        pass
    try:
        ymd = d.replace("-", "")
        js = requests.get(f"https://www.twse.com.tw/rwd/zh/afterTrading/FMTQIK?date={ymd}&response=json",
                          headers=UA, timeout=20).json()
        roc = f"{int(ymd[:4]) - 1911}/{ymd[4:6]}/{ymd[6:]}"
        for row in js.get("data", []) or []:
            if row[0].strip() == roc:
                return float(row[4].replace(",", ""))
    except Exception as e:  # noqa: BLE001
        print(f"[warn] TAIEX {d}: {e}", file=sys.stderr)
    return None


# ---------------------------------------------------------------- commands
def cmd_init(a):
    if LEDGER.exists() and not a.force:
        sys.exit("ledger.json 已存在；要重建請加 --force")
    led = dict(initial_cash=a.cash, cash=a.cash, start_date=a.start, positions={}, orders=[],
               trades=[], equity=[], next_id=1, benchmark={})
    save(led)
    print(f"初始化完成：現金 {a.cash:,.0f}，起始日 {a.start}")


def cmd_order(a):
    led = load()
    side = a.side.upper()
    if side not in ("BUY", "SELL"):
        sys.exit("side 必須是 BUY/SELL")
    o = dict(id=led["next_id"], date=a.date, code=a.code, side=side, shares=int(a.shares),
             limit=a.limit, stop=a.stop, target=a.target, note=a.note or "", status="PENDING")
    led["orders"].append(o)
    led["next_id"] += 1
    save(led)
    print(f"委託 #{o['id']} {side} {a.code} {a.shares} 股 @ {a.limit or '開盤市價'} ({a.date})")


def cmd_cancel(a):
    led = load()
    for o in led["orders"]:
        if o["id"] == a.order_id and o["status"] == "PENDING":
            o["status"] = "CANCELLED"
            save(led)
            print(f"已取消 #{a.order_id}")
            return
    sys.exit("找不到可取消的委託")


def _buy(led, code, shares, px, d, o=None):
    amt = px * shares
    f = fee(amt, shares)
    if amt + f > led["cash"] + 1e-6:
        return f"現金不足（需要 {amt + f:,.0f}，現有 {led['cash']:,.0f}）"
    led["cash"] -= amt + f
    p = led["positions"].get(code)
    if p:
        tot = p["avg_cost"] * p["shares"] + amt + f
        p["shares"] += shares
        p["avg_cost"] = tot / p["shares"]
    else:
        p = led["positions"][code] = dict(shares=shares, avg_cost=(amt + f) / shares, opened=d,
                                          stop=None, target=None, last_buy=d)
    p["last_buy"] = d
    if o:
        if o.get("stop"):
            p["stop"] = o["stop"]
        if o.get("target"):
            p["target"] = o["target"]
    led["trades"].append(dict(date=d, code=code, side="BUY", shares=shares, price=px, fee=f, tax=0,
                              note=(o or {}).get("note", "")))
    return None


def _sell(led, code, shares, px, d, note=""):
    p = led["positions"].get(code)
    if not p or p["shares"] < shares:
        return "持股不足"
    amt = px * shares
    daytrade = p.get("last_buy") == d
    f, t = fee(amt, shares), tax(amt, code, daytrade)
    pnl = amt - f - t - p["avg_cost"] * shares
    led["cash"] += amt - f - t
    p["shares"] -= shares
    if p["shares"] == 0:
        del led["positions"][code]
    led["trades"].append(dict(date=d, code=code, side="SELL", shares=shares, price=px, fee=f, tax=t,
                              pnl=round(pnl), daytrade=daytrade, note=note))
    return None


def cmd_settle(a):
    led = load()
    d = a.date
    log = []
    # 1) 撮合當日委託（先賣後買，釋出現金）
    todays = [o for o in led["orders"] if o["status"] == "PENDING" and o["date"] == d]
    todays.sort(key=lambda o: 0 if o["side"] == "SELL" else 1)
    for o in todays:
        k = ohlc(o["code"], d)
        if not k:
            o["status"] = "NO_DATA"
            log.append(f"#{o['id']} {o['code']} 無當日行情（停牌或資料未出），未成交")
            continue
        odd = o["shares"] % 1000 != 0
        slip = SLIPPAGE if odd else 0.0
        px = None
        lk = locked(k)
        if (o["side"] == "BUY" and lk == "UP") or (o["side"] == "SELL" and lk == "DOWN"):
            o["status"] = "EXPIRED"
            log.append(f"#{o['id']} {o['side']} {o['code']} 當日一價鎖{'漲' if lk == 'UP' else '跌'}停，視為排不到未成交")
            continue
        if o["side"] == "BUY":
            if o["limit"] is None:
                px = round_tick(k["open"] * (1 + SLIPPAGE), up=True)
            elif k["open"] <= o["limit"]:
                px = k["open"]
            elif k["low"] <= o["limit"] * (1 - slip):
                px = o["limit"]
            err = _buy(led, o["code"], o["shares"], px, d, o) if px else None
        else:
            if o["limit"] is None:
                px = round_tick(k["open"] * (1 - SLIPPAGE), up=False)
            elif k["open"] >= o["limit"]:
                px = k["open"]
            elif k["high"] >= o["limit"] * (1 + slip):
                px = o["limit"]
            err = _sell(led, o["code"], o["shares"], px, d, o.get("note", "")) if px else None
        if px and not err:
            o.update(status="FILLED", fill_price=px)
            log.append(f"#{o['id']} {o['side']} {o['code']} {o['shares']} 股 成交 @ {px}")
        elif err:
            o.update(status="REJECTED", reason=err)
            log.append(f"#{o['id']} {o['code']} 被拒：{err}")
        else:
            o["status"] = "EXPIRED"
            log.append(f"#{o['id']} {o['side']} {o['code']} @ {o['limit']} 未成交（當日 L {k['low']} / H {k['high']}）")
    # 2) 停損 / 停利（不檢查當天才買進的部位的開盤，但盤中仍檢查）
    for code, p in list(led["positions"].items()):
        k = ohlc(code, d)
        if not k:
            continue
        s, tg = p.get("stop"), p.get("target")
        if p.get("force_exit") and locked(k) != "DOWN":
            px = round_tick(k["open"] * (1 - SLIPPAGE), up=False)
            _sell(led, code, p["shares"], px, d, "停損(前日鎖跌停延續)")
            log.append(f"{code} 前日停損未能出場，今日開盤出場 @ {px}")
            continue
        if s and k["low"] <= s and locked(k) == "DOWN":
            log.append(f"{code} 跌破停損 {s} 但鎖跌停賣不掉，明日開盤續處理")
            p["force_exit"] = True
        elif s and k["low"] <= s:
            px = k["open"] if (k["open"] <= s and p.get("last_buy") != d) else round_tick(s * (1 - SLIPPAGE), up=False)
            _sell(led, code, p["shares"], px, d, "停損")
            log.append(f"{code} 觸發停損 {s}，出場 @ {px}")
        elif tg and k["high"] >= tg:
            half = p["shares"] // 2 or p["shares"]
            _sell(led, code, half, tg, d, "停利(半)")
            if code in led["positions"]:
                led["positions"][code]["target"] = None
                led["positions"][code]["stop"] = max(s or 0, round(led["positions"][code]["avg_cost"], 2))
            log.append(f"{code} 觸及目標 {tg}，賣出 {half} 股，剩餘部位停損上移至成本")
    # 3) 計算淨值
    mv = 0.0
    for code, p in led["positions"].items():
        k = ohlc(code, d)
        p["last_close"] = k["close"] if k else p.get("last_close", p["avg_cost"])
        mv += p["last_close"] * p["shares"]
    eq = led["cash"] + mv
    ix = index_close(d)
    if ix and "taiex_start" not in led["benchmark"]:
        led["benchmark"]["taiex_start"] = ix
    led["equity"] = [e for e in led["equity"] if e["date"] != d]
    led["equity"].append(dict(date=d, cash=round(led["cash"]), market_value=round(mv), equity=round(eq),
                              taiex=ix))
    save(led)
    print("\n".join(log) or "（今日無委託）")
    _print_status(led)


def _print_status(led):
    eq = led["equity"][-1] if led["equity"] else None
    print("\n=== 部位 ===")
    for code, p in led["positions"].items():
        lc = p.get("last_close", p["avg_cost"])
        print(f"{code}: {p['shares']} 股 成本 {p['avg_cost']:.2f} 收 {lc} 損益 {((lc / p['avg_cost']) - 1) * 100:+.2f}%"
              f" 停損 {p.get('stop')} 目標 {p.get('target')}")
    print(f"現金 {led['cash']:,.0f}")
    if eq:
        ret = eq["equity"] / led["initial_cash"] - 1
        bm = led["benchmark"].get("taiex_start")
        bret = (eq["taiex"] / bm - 1) if (bm and eq.get("taiex")) else None
        print(f"淨值 {eq['equity']:,} ({ret * 100:+.2f}%)" + (f"｜加權指數同期 {bret * 100:+.2f}%" if bret is not None else ""))
    pend = [o for o in led["orders"] if o["status"] == "PENDING"]
    if pend:
        print("待撮合委託：", ", ".join(f"#{o['id']} {o['side']} {o['code']} {o['shares']}@{o['limit']} ({o['date']})" for o in pend))


def cmd_status(a):
    _print_status(load())


def main():
    today = date.today().isoformat()
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("init"); p.add_argument("--cash", type=float, default=100000)
    p.add_argument("--start", default=today); p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_init)
    p = sp.add_parser("order"); p.add_argument("side"); p.add_argument("code"); p.add_argument("shares", type=int)
    p.add_argument("--limit", type=float); p.add_argument("--stop", type=float); p.add_argument("--target", type=float)
    p.add_argument("--note"); p.add_argument("--date", default=today); p.set_defaults(fn=cmd_order)
    p = sp.add_parser("cancel"); p.add_argument("order_id", type=int); p.set_defaults(fn=cmd_cancel)
    p = sp.add_parser("settle"); p.add_argument("--date", default=today); p.set_defaults(fn=cmd_settle)
    p = sp.add_parser("status"); p.set_defaults(fn=cmd_status)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
