"""台股模擬盤引擎（現股，上市股票/ETF，可零股、可現股當沖）。

用法：
  python engine.py init [--cash 100000] [--start 2026-11-26]
  python engine.py order BUY 2330 10 --limit 1000 [--stop 920] [--target 1150] [--note "..."] [--date 2026-11-26]
  python engine.py order SELL 2330 10 [--limit 1100] [--date ...]      # 不給 --limit = 開盤市價
  python engine.py cancel <order_id>
  python engine.py settle [--date 2026-11-26]   # 用當日 OHLC 撮合委託、檢查停損/停利、記錄淨值
  python engine.py status

撮合規則（保守近似）：
  - 優先用 Yahoo 5 分鐘 K 依時間順序模擬；拿不到才用日 K（視為單一根 K）。
  - 委託自 --after（預設 09:00）起生效；cancel --at 之後不再成交（之前已成交不受影響）。
  - 買進限價 L：該根 K 開盤 <= L 以開盤價成交；否則最低 <= L 以 L 成交（零股需再低 0.2%）。
  - 賣出限價對稱；市價單以生效後第一根 K 開盤價 ± 0.2% 滑價成交。
  - 停損 S：該根 K 最低 <= S 即出場，價格為 min(開盤, S) 減 0.2% 滑價；觸及目標先賣一半、停損移到成本。
    同一根 K 同時觸及停損與目標時假設先停損；當日新買進的部位從下一根 K 才開始檢查。
  - setstop --at：盤中調整停損，只對之後的 K 生效。
  - 一價鎖漲停買不到、鎖跌停賣不掉（停損順延至次日開盤）。
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


def roundtrip_cost(code: str, price: float, shares: int, daytrade: bool = False) -> float:
    amt = price * shares
    return fee(amt, shares) * 2 + tax(amt, code, daytrade)


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
    if side == "BUY" and not (a.strategy and a.note and a.invalid):
        sys.exit("買進委託必須提供 --strategy、--note（理由）與 --invalid（失效條件）")
    o = dict(id=led["next_id"], date=a.date, code=a.code, side=side, shares=int(a.shares),
             limit=a.limit, stop=a.stop, target=a.target, note=a.note or "", status="PENDING",
             strategy=a.strategy or (led["positions"].get(a.code, {}).get("strategy")),
             invalid=a.invalid or "", prob=a.prob, after=a.after or "09:00")
    led["orders"].append(o)
    led["next_id"] += 1
    save(led)
    print(f"委託 #{o['id']} {side} {a.code} {a.shares} 股 @ {a.limit or '市價'} ({a.date} {o['after']} 起) [{o['strategy']}]")
    if a.limit:
        c = roundtrip_cost(a.code, a.limit, int(a.shares))
        print(f"  來回成本約 {c:,.0f} 元（{c / (a.limit * a.shares) * 100:.2f}%）"
              + (f"；目標報酬 {(a.target / a.limit - 1) * 100:+.1f}%" if a.target and side == "BUY" else "")
              + (f"；停損風險 {(a.stop / a.limit - 1) * 100:+.1f}%" if a.stop and side == "BUY" else ""))


def cmd_cancel(a):
    led = load()
    for o in led["orders"]:
        if o["id"] == a.order_id and o["status"] == "PENDING":
            if a.at:  # 盤中取消：只取消 a.at 之後尚未成交的部分（結算時判斷）
                o["cancel_at"] = a.at
                print(f"#{a.order_id} 設定於 {a.at} 取消（若在此之前已成交則不受影響）")
            else:
                o["status"] = "CANCELLED"
                print(f"已取消 #{a.order_id}")
            if a.reason:
                o["cancel_reason"] = a.reason
            save(led)
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
                                          stop=None, target=None, last_buy=d,
                                          strategy=(o or {}).get("strategy"), thesis=(o or {}).get("note", ""),
                                          invalid=(o or {}).get("invalid", ""), prob=(o or {}).get("prob"),
                                          init_stop=(o or {}).get("stop"))
    p["last_buy"] = d
    if o:
        if o.get("stop"):
            p["stop"] = o["stop"]
        if o.get("target"):
            p["target"] = o["target"]
    led["trades"].append(dict(date=d, code=code, side="BUY", shares=shares, price=px, fee=f, tax=0,
                              strategy=(o or {}).get("strategy"), note=(o or {}).get("note", "")))
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
                              pnl=round(pnl), daytrade=daytrade, note=note, strategy=p.get("strategy"),
                              prob=p.get("prob"), opened=p.get("opened"),
                              ret=round(px / p["avg_cost"] - 1, 4)))
    return None


_bars_cache: dict[tuple[str, str], list | None] = {}


def bars(code: str, d: str) -> list | None:
    """當日 5 分鐘 K（Yahoo，約保留 60 天）：[(HH:MM, o, h, l, c), ...]；拿不到回傳 None。"""
    key = (code, d)
    if key in _bars_cache:
        return _bars_cache[key]
    res = None
    try:
        import yfinance as yf
        h = yf.Ticker(f"{code}.TW").history(period="60d", interval="5m")
        h = h[h.index.strftime("%Y-%m-%d") == d]
        if len(h):
            res = [(i.strftime("%H:%M"), float(r.Open), float(r.High), float(r.Low), float(r.Close))
                   for i, r in h.iterrows() if r.High > 0]
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 5m {code} {d}: {e}", file=sys.stderr)
    _bars_cache[key] = res or None
    return _bars_cache[key]


def _day_bars(code: str, d: str, k: dict) -> tuple[list, bool]:
    """優先 5 分鐘 K；沒有就用日 K 當成單一根（退化為日 K 邏輯）。回傳 (bars, is_intraday)。"""
    b = bars(code, d)
    if b:
        return b, True
    return [("09:00", k["open"], k["high"], k["low"], k["close"])], False


def exdiv(code: str, d: str) -> dict | None:
    """當日除權息結果（FinMind TaiwanStockDividendResult）：{'cash': 現金股利, 'kind': '息'/'權'/'權息'}。"""
    try:
        js = requests.get("https://api.finmindtrade.com/api/v4/data",
                          params=dict(dataset="TaiwanStockDividendResult", data_id=code, start_date=d, end_date=d),
                          timeout=20).json()
        for r in js.get("data", []):
            if r["date"] == d:
                return dict(amount=float(r["stock_and_cache_dividend"]), kind=r["stock_or_cache_dividend"],
                            before=r["before_price"], after=r["after_price"])
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 除權息 {code} {d}: {e}", file=sys.stderr)
    return None


def _apply_dividends(led, d, log):
    """除息日：現金股利入帳（實際約一個月後撥付，這裡簡化為當日），停損/目標同步下調；影子部位價位一併調整。
    除權（配股）需人工處理，僅提示。"""
    codes = set(led["positions"]) | {s_["code"] for s_ in led.get("shadow", []) if s_["status"] in ("WAIT", "OPEN")}
    for code in codes:
        x = exdiv(code, d)
        if not x:
            continue
        if x["kind"] != "息":
            log.append(f"⚠️ {code} {d} 除權（{x['kind']}），配股需人工調整股數")
            continue
        cash_div = round(x["before"] - x["after"], 4) or x["amount"]
        p = led["positions"].get(code)
        if p:
            amt = math.floor(cash_div * p["shares"])
            led["cash"] += amt
            led["trades"].append(dict(date=d, code=code, side="DIV", shares=p["shares"], price=cash_div, fee=0, tax=0,
                                      pnl=amt, strategy=p.get("strategy"), note="現金股利"))
            for f in ("stop", "target", "init_stop"):
                if p.get(f):
                    p[f] = round(p[f] - cash_div, 2)
            log.append(f"{code} 除息 {cash_div:.2f} 元，股利 {amt} 元入帳，停損/目標同步下調")
        for s_ in led.get("shadow", []):
            if s_["code"] == code and s_["status"] in ("WAIT", "OPEN"):
                for f in ("entry", "stop", "target", "fill"):
                    if s_.get(f):
                        s_[f] = round(s_[f] - cash_div, 2)


def cmd_settle(a):
    """以 5 分鐘 K 依時間順序模擬：委託（含 --after 生效時間、cancel --at）、停損/停利、盤中改停損。"""
    led = load()
    d = a.date
    log = []
    _apply_dividends(led, d, log)
    todays = [o for o in led["orders"] if o["status"] == "PENDING" and o["date"] == d]
    codes = {o["code"] for o in todays} | set(led["positions"])
    daily, series, intraday = {}, {}, {}
    for c in codes:
        k = ohlc(c, d)
        if k:
            daily[c] = k
            series[c], intraday[c] = _day_bars(c, d, k)
    for o in todays:
        if o["code"] not in daily:
            o["status"] = "NO_DATA"
            log.append(f"#{o['id']} {o['code']} 無當日行情（停牌或資料未出），未成交")
        elif (o["side"] == "BUY" and locked(daily[o["code"]]) == "UP") or \
             (o["side"] == "SELL" and locked(daily[o["code"]]) == "DOWN"):
            o["status"] = "EXPIRED"
            log.append(f"#{o['id']} {o['side']} {o['code']} 當日一價鎖停板，視為排不到未成交")
    fill_time: dict[str, str] = {}
    timeline = sorted({b[0] for c in series for b in series[c]})
    for t in timeline:
        # 1) 委託：先賣後買
        live = [o for o in todays if o["status"] == "PENDING" and t >= o.get("after", "09:00")
                and not (o.get("cancel_at") and t >= o["cancel_at"])]
        live.sort(key=lambda o: 0 if o["side"] == "SELL" else 1)
        for o in live:
            bar = next((b for b in series.get(o["code"], []) if b[0] == t), None)
            if not bar:
                continue
            _, bo, bh, bl, _bc = bar
            slip = SLIPPAGE if o["shares"] % 1000 else 0.0
            px = None
            if o["side"] == "BUY":
                if o["limit"] is None:
                    px = round_tick(bo * (1 + SLIPPAGE), up=True)
                elif bo <= o["limit"]:
                    px = round(bo, 2)
                elif bl <= o["limit"] * (1 - slip):
                    px = o["limit"]
                err = _buy(led, o["code"], o["shares"], px, d, o) if px else None
            else:
                if o["limit"] is None:
                    px = round_tick(bo * (1 - SLIPPAGE), up=False)
                elif bo >= o["limit"]:
                    px = round(bo, 2)
                elif bh >= o["limit"] * (1 + slip):
                    px = o["limit"]
                err = _sell(led, o["code"], o["shares"], px, d, o.get("note", "")) if px else None
            if px and not err:
                o.update(status="FILLED", fill_price=px, fill_time=t if intraday[o["code"]] else None)
                fill_time[o["code"]] = t
                log.append(f"#{o['id']} {o['side']} {o['code']} {o['shares']} 股 成交 @ {px}"
                           + (f"（{t}）" if intraday[o["code"]] else "（日K近似）"))
            elif err:
                o.update(status="REJECTED", reason=err)
                log.append(f"#{o['id']} {o['code']} 被拒：{err}")
        # 2) 停損 / 停利（同一根 K 先停損後停利；當日新買進的部位從下一根 K 才開始檢查）
        for code, p in list(led["positions"].items()):
            bar = next((b for b in series.get(code, []) if b[0] == t), None)
            if not bar or (code in fill_time and t <= fill_time[code] and intraday[code]):
                continue
            _, bo, bh, bl, _bc = bar
            ps = p.get("pending_stop")
            if ps and (ps["date"] < d or (ps["date"] == d and t >= ps["at"])):
                p["stop"] = ps["stop"]
                p.pop("pending_stop")
                log.append(f"{code} 停損調整為 {p['stop']}（{ps['date']} {ps['at']} 生效）")
            s, tg = p.get("stop"), p.get("target")
            lk = locked(daily[code])
            if p.get("force_exit") and lk != "DOWN":
                px = round_tick(bo * (1 - SLIPPAGE), up=False)
                _sell(led, code, p["shares"], px, d, "停損(前日鎖跌停延續)")
                log.append(f"{code} 前日停損未能出場，今日 {t} 出場 @ {px}")
            elif s and bl <= s and lk == "DOWN":
                if not p.get("force_exit"):
                    log.append(f"{code} 跌破停損 {s} 但鎖跌停賣不掉，明日開盤續處理")
                p["force_exit"] = True
            elif s and bl <= s:
                px = round_tick((bo if bo <= s else s) * (1 - SLIPPAGE), up=False)
                _sell(led, code, p["shares"], px, d, "停損")
                log.append(f"{code} 觸發停損 {s}，{t if intraday[code] else ''} 出場 @ {px}")
            elif tg and bh >= tg:
                half = p["shares"] // 2 or p["shares"]
                tpx = round(bo, 2) if bo >= tg else tg
                _sell(led, code, half, tpx, d, "停利(半)")
                if code in led["positions"]:
                    led["positions"][code]["target"] = None
                    led["positions"][code]["stop"] = max(s or 0, round(led["positions"][code]["avg_cost"], 2))
                log.append(f"{code} 觸及目標 {tg}，賣出 {half} 股 @ {tpx}，剩餘停損上移至成本")
    for o in todays:
        if o["status"] == "PENDING":
            if o.get("cancel_at"):
                o["status"] = "CANCELLED"
                log.append(f"#{o['id']} {o['code']} 於 {o['cancel_at']} 前未成交，已取消")
            else:
                k = daily.get(o["code"])
                o["status"] = "EXPIRED"
                log.append(f"#{o['id']} {o['side']} {o['code']} @ {o['limit']} 未成交"
                           + (f"（當日 L {k['low']} / H {k['high']}）" if k else ""))
    for code, p in led["positions"].items():  # 盤後才送出的改停損
        ps = p.get("pending_stop")
        if ps and ps["date"] <= d:
            p["stop"] = ps["stop"]
            p.pop("pending_stop")
    _settle_shadow(led, d, log)
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
    prev = led["equity"][-1] if led["equity"] else None
    # 同曝險基準：昨日持股比重 × 今日加權指數報酬，累積起來，衡量「選股/擇時」是否勝過單純的大盤曝險
    adj = prev.get("bench_adj", 1.0) if prev else 1.0
    if prev and prev.get("taiex") and ix and prev["equity"]:
        adj *= 1 + (prev["market_value"] / prev["equity"]) * (ix / prev["taiex"] - 1)
    led["equity"].append(dict(date=d, cash=round(led["cash"]), market_value=round(mv), equity=round(eq),
                              taiex=ix, bench_adj=round(adj, 6)))
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
        print(f"淨值 {eq['equity']:,} ({ret * 100:+.2f}%)" + (f"｜加權指數同期 {bret * 100:+.2f}%" if bret is not None else "")
              + (f"｜同曝險基準 {(eq.get('bench_adj', 1) - 1) * 100:+.2f}%" if eq.get("bench_adj") else ""))
    pend = [o for o in led["orders"] if o["status"] == "PENDING"]
    if pend:
        print("待撮合委託：", ", ".join(f"#{o['id']} {o['side']} {o['code']} {o['shares']}@{o['limit']} ({o['date']})" for o in pend))


def cmd_setstop(a):
    """調整停損（只能上移，除非 --force）。盤中決策用 --at HH:MM，結算時只對之後的 K 棒生效。"""
    led = load()
    p = led["positions"].get(a.code)
    if not p:
        sys.exit("無此持股")
    if p.get("stop") and a.stop < p["stop"] and not a.force:
        sys.exit(f"停損只能上移（目前 {p['stop']}）；確定要下移請加 --force 並在報告說明")
    if a.at:
        p["pending_stop"] = dict(date=a.date, at=a.at, stop=a.stop)
    else:
        p["stop"] = a.stop
    save(led)
    print(f"{a.code} 停損 → {a.stop}" + (f"（{a.date} {a.at} 起生效）" if a.at else ""))


def cmd_shadow(a):
    """影子部位：記錄『看到但沒做』或規則訊號的假想交易，不占資金，用來擴大樣本、降低選擇偏誤。"""
    led = load()
    led.setdefault("shadow", []).append(dict(date=a.date, code=a.code, entry=a.entry, stop=a.stop, target=a.target,
                                             strategy=a.strategy, note=a.note or "", prob=a.prob, status="WAIT",
                                             expire=a.expire))
    save(led)
    print(f"影子 {a.code} 進場 {a.entry} 停損 {a.stop} 目標 {a.target} [{a.strategy}]")


def _settle_shadow(led, d, log):
    for s_ in led.get("shadow", []):
        if s_["status"] in ("WIN", "LOSS", "EXPIRED") or s_["date"] > d:
            continue
        k = ohlc(s_["code"], d)
        if not k:
            continue
        if s_["status"] == "WAIT":
            if locked(k) == "UP" or k["low"] > s_["entry"]:
                if s_.get("expire") and d >= s_["expire"]:
                    s_["status"] = "EXPIRED"
                continue
            s_.update(status="OPEN", fill=min(k["open"], s_["entry"]), filled=d)
        if s_["status"] == "OPEN":
            if k["low"] <= s_["stop"]:
                s_.update(status="LOSS", exit=s_["stop"], closed=d)
            elif k["high"] >= s_["target"] and d != s_["filled"]:
                s_.update(status="WIN", exit=s_["target"], closed=d)
            if s_["status"] in ("WIN", "LOSS"):
                s_["r"] = round((s_["exit"] - s_["fill"]) / (s_["fill"] - s_["stop"]), 2)
                log.append(f"影子 {s_['code']} [{s_['strategy']}] {s_['status']} R={s_['r']}")


def cmd_stats(a):
    led = load()
    agg: dict[str, dict] = {}
    for t in led["trades"]:
        g = agg.setdefault(t.get("strategy") or "未標記", dict(n=0, win=0, pnl=0, cost=0))
        g["cost"] += t["fee"] + t["tax"]
        if t["side"] == "SELL":
            g["n"] += 1
            g["win"] += t["pnl"] > 0
            g["pnl"] += t["pnl"]
    for code, p in led["positions"].items():
        g = agg.setdefault(p.get("strategy") or "未標記", dict(n=0, win=0, pnl=0, cost=0))
        g["open"] = g.get("open", 0) + round((p.get("last_close", p["avg_cost"]) - p["avg_cost"]) * p["shares"])
    print(f"{'策略':<14}{'平倉筆數':>6}{'勝率':>8}{'已實現(淨)':>12}{'未實現':>10}{'累計費稅':>10}")
    for k, g in agg.items():
        wr = f"{g['win'] / g['n'] * 100:.0f}%" if g["n"] else "-"
        print(f"{k:<14}{g['n']:>6}{wr:>8}{g['pnl']:>12,}{g.get('open', 0):>10,}{g['cost']:>10,}")
    sh = [x for x in led.get("shadow", []) if x["status"] in ("WIN", "LOSS")]
    if sh:
        print("\n影子部位（含實單以外的訊號）：")
        by: dict[str, list] = {}
        for x in sh:
            by.setdefault(x["strategy"], []).append(x["r"])
        for k, rs in by.items():
            print(f"  {k}: {len(rs)} 筆, 平均 R {sum(rs) / len(rs):+.2f}, 勝率 {sum(r > 0 for r in rs) / len(rs) * 100:.0f}%")
    pr = [(t["prob"], 1 if t["pnl"] > 0 else 0) for t in led["trades"] if t["side"] == "SELL" and t.get("prob") is not None]
    pr += [(x["prob"], 1 if x["status"] == "WIN" else 0) for x in sh if x.get("prob") is not None]
    if pr:
        brier = sum((p - o) ** 2 for p, o in pr) / len(pr)
        print(f"\n機率校準：{len(pr)} 筆，Brier {brier:.3f}（0.25 = 擲硬幣水準，越低越好）")


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
    p.add_argument("--note", help="交易理由"); p.add_argument("--strategy", help="策略標籤")
    p.add_argument("--invalid", help="失效條件：什麼情況代表判斷錯了")
    p.add_argument("--prob", type=float, help="事前機率：先到目標而非停損的機率(0-1)，用於校準")
    p.add_argument("--after", help="生效時間 HH:MM（盤中決策用，預設 09:00）")
    p.add_argument("--date", default=today); p.set_defaults(fn=cmd_order)
    p = sp.add_parser("cancel"); p.add_argument("order_id", type=int)
    p.add_argument("--at", help="盤中取消時間 HH:MM"); p.add_argument("--reason"); p.set_defaults(fn=cmd_cancel)
    p = sp.add_parser("setstop"); p.add_argument("code"); p.add_argument("stop", type=float)
    p.add_argument("--at", help="盤中生效時間 HH:MM"); p.add_argument("--date", default=today)
    p.add_argument("--force", action="store_true"); p.set_defaults(fn=cmd_setstop)
    p = sp.add_parser("settle"); p.add_argument("--date", default=today); p.set_defaults(fn=cmd_settle)
    p = sp.add_parser("status"); p.set_defaults(fn=cmd_status)
    p = sp.add_parser("stats"); p.set_defaults(fn=cmd_stats)
    p = sp.add_parser("shadow"); p.add_argument("code"); p.add_argument("--entry", type=float, required=True)
    p.add_argument("--stop", type=float, required=True); p.add_argument("--target", type=float, required=True)
    p.add_argument("--strategy", required=True); p.add_argument("--note"); p.add_argument("--prob", type=float)
    p.add_argument("--expire", help="未進場的失效日"); p.add_argument("--date", default=today)
    p.set_defaults(fn=cmd_shadow)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
