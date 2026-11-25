#!/usr/bin/env python3
"""TWSE swing screener. Run fetch_twse.py first.

Usage: python screen.py [YYYYMMDD] [SELLOFF_BASE_YYYYMMDD]
Outputs: screen_<date>.csv (all liquid names, ranked) + console top lists.

Data sources
- Official TWSE MI_INDEX (last 22 trading days): OHLCV, turnover, ATR, MA10/20, 20d high.
- Official TWSE T86 (last 10 days): foreign (incl. foreign dealer) + investment trust net buy.
- Yahoo (yfinance, auto-adjusted) for 60-day history (MA60, 60d return, RS) and to derive
  dividend-adjustment factors for the TWSE bars. Latest bar always = TWSE official.

Composite score (cross-sectional percentile ranks, 0-100, computed separately for stocks and ETFs):
  Trend 35%   : 10% rel.ret60 vs TAIEX, 10% rel.ret20 vs TAIEX, 7.5% close/MA20, 7.5% close/MA60
  Flow  25%   : (foreign + trust) 10-day net buy / 20-day avg volume
  Resilience 20%: return since SELLOFF_BASE close minus TAIEX return
  Risk  20%   : 10% low ATR14%, 10% close near 20-day high
  Penalties   : -10 if close/MA20 > +15% (over-extended), -10 if ATR% > 6%, -5 if 注意股 today.
  Excluded    : 處置股 (active on next trading day), 變更交易/全額交割 (TWT85U), 20d avg turnover < NT$300m.
"""
import json, os, sys, glob, math, re
import numpy as np
import pandas as pd
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
TARGET = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob(os.path.join(RAW, "mi_*.json")))[-1][-13:-5]
SELLOFF_BASE = sys.argv[2] if len(sys.argv) > 2 else "20261118"
MIN_TURNOVER = 300e6
LEVERAGED = re.compile(r"(?:正2|反1|反向|槓桿|2X|-1X)")


def num(x):
    try:
        x = str(x).replace(",", "").strip()
        return float(x) if x not in ("", "--", "---", "----") else np.nan
    except Exception:
        return np.nan


def load_mi(d):
    j = json.load(open(os.path.join(RAW, f"mi_{d}.json")))
    t = [t for t in j["tables"] if "每日收盤行情" in (t.get("title") or "")][0]
    rows = []
    for r in t["data"]:
        rows.append(dict(date=d, code=r[0].strip(), name=r[1].strip(), vol=num(r[2]), value=num(r[4]),
                         open=num(r[5]), high=num(r[6]), low=num(r[7]), close=num(r[8])))
    return pd.DataFrame(rows)


def load_t86(d):
    j = json.load(open(os.path.join(RAW, f"t86_{d}.json")))
    f = j["fields"]
    i_for = f.index("外陸資買賣超股數(不含外資自營商)")
    i_fd = f.index("外資自營商買賣超股數")
    i_it = f.index("投信買賣超股數")
    rows = [dict(code=r[0].strip(), foreign=num(r[i_for]) + num(r[i_fd]), trust=num(r[i_it])) for r in j["data"]]
    return pd.DataFrame(rows).assign(date=d)


def tick(p, etf=False):
    if etf:
        return 0.01 if p < 50 else 0.05
    for lim, t in [(10, 0.01), (50, 0.05), (100, 0.1), (500, 0.5), (1000, 1.0)]:
        if p < lim:
            return t
    return 5.0


def rnd(p, etf=False, mode="near"):
    t = tick(p, etf)
    f = {"near": round, "down": math.floor, "up": math.ceil}[mode]
    return round(f(p / t) * t, 2)


def main():
    cal = json.load(open(os.path.join(RAW, "calendar.json")))
    cal = [d for d in cal if d <= TARGET]
    mi_days = [d for d in cal if os.path.exists(os.path.join(RAW, f"mi_{d}.json"))][-22:]
    t86_days = [d for d in cal if os.path.exists(os.path.join(RAW, f"t86_{d}.json"))][-10:]
    assert mi_days[-1] == TARGET, "latest MI_INDEX missing"
    gaps = []
    if len(t86_days) < 10 or t86_days[-1] != TARGET:
        gaps.append(f"T86 days available: {t86_days}")
    mi = pd.concat([load_mi(d) for d in mi_days])
    last = mi[mi.date == TARGET].set_index("code")

    # ---- universe ----
    is_stock = last.index.str.fullmatch(r"[1-9]\d{3}")
    is_etf = last.index.str.match(r"^00\d{2,4}[A-Z]?$") & ~last.index.str.endswith(("L", "R")) \
        & ~last["name"].str.contains(LEVERAGED)
    is_bond = last.index.str.endswith("B")
    is_tib = last["name"].str.contains(r"創$|-創")  # 創新板: qualified investors only
    last = last.assign(kind=np.where(is_stock & ~is_tib, "stock",
                                     np.where(is_etf & is_bond, "etf_bond", np.where(is_etf, "etf", ""))))
    last = last[last.kind != ""]
    d20 = mi_days[-20:]
    turn = mi[mi.date.isin(d20)].groupby("code")["value"].mean()
    vol20 = mi[mi.date.isin(d20)].groupby("code")["vol"].mean()
    last = last.assign(turn20=turn.reindex(last.index), vol20=vol20.reindex(last.index))
    liquid = last[(last.turn20 >= MIN_TURNOVER) & last.close.notna()].copy()

    # ---- status lists ----
    punish = json.load(open(os.path.join(RAW, f"punish_{TARGET}.json")))
    nextday = cal[-1] if False else None
    disp = {}
    for r in punish["data"]:
        code = r[2].strip()
        per = r[6]
        m = re.findall(r"(\d+)/(\d+)/(\d+)", per)
        if len(m) == 2:
            end = f"{int(m[1][0]) + 1911}{int(m[1][1]):02d}{int(m[1][2]):02d}"
            if end > TARGET:  # still active on next trading day
                disp[code] = f"處置至{end[4:]}"
    try:
        for r in json.load(open(os.path.join(RAW, f"punish_oa_{TARGET}.json"))):
            m = re.findall(r"(\d+)/(\d+)/(\d+)", r.get("DispositionPeriod", ""))
            if len(m) == 2:
                end = f"{int(m[1][0]) + 1911}{int(m[1][1]):02d}{int(m[1][2]):02d}"
                if end > TARGET:
                    disp.setdefault(r["Code"].strip(), f"處置至{end[4:]}")
    except FileNotFoundError:
        gaps.append("openapi punish missing")
    chg = {r["Code"].strip() for r in json.load(open(os.path.join(RAW, f"twt85u_{TARGET}.json")))}
    notice = {r[1].strip(): r[4] for r in json.load(open(os.path.join(RAW, f"notice_{TARGET}.json")))["data"]}
    try:
        notetrans = {r["Code"].strip() for r in json.load(open(os.path.join(RAW, f"notetrans_{TARGET}.json")))}
    except Exception:
        notetrans = set()
    liquid["excluded"] = ""
    for c in liquid.index:
        if c in disp:
            liquid.at[c, "excluded"] = disp[c]
        elif c in chg:
            liquid.at[c, "excluded"] = "變更交易/全額交割"
    liquid["attention"] = [("注意股" if c in notice else "") + ("/注意累計近處置" if c in notetrans else "")
                           for c in liquid.index]

    # ---- Yahoo history for 60d metrics + adjustment factors ----
    tick_map = {c: f"{c}.TW" for c in liquid.index}
    yh = yf.download(list(tick_map.values()) + ["^TWII"], period="9mo", auto_adjust=True,
                     progress=False, threads=True, group_by="column")
    ycl = yh["Close"].copy()
    ycl.index = ycl.index.strftime("%Y%m%d")
    ycl = ycl[ycl.index <= TARGET]
    twii = ycl["^TWII"].dropna()
    # TAIEX official close for target from MI_INDEX index table
    j = json.load(open(os.path.join(RAW, f"mi_{TARGET}.json")))
    for t in j["tables"]:
        for r in t.get("data", []) or []:
            if str(r[0]).strip() == "發行量加權股價指數":
                twii.loc[TARGET] = num(r[1])
    twii = twii.sort_index()

    def tw_ret(n):
        return twii.iloc[-1] / twii.iloc[-1 - n] - 1

    tw_r20, tw_r60 = tw_ret(20), tw_ret(60)
    tw_sell = twii.iloc[-1] / twii.loc[SELLOFF_BASE] - 1

    mi_piv = {k: mi.pivot_table(index="date", columns="code", values=k) for k in ["open", "high", "low", "close", "vol"]}
    out = []
    for c in liquid.index:
        y = ycl[tick_map[c]].dropna() if tick_map[c] in ycl else pd.Series(dtype=float)
        tw = pd.DataFrame({k: mi_piv[k][c] for k in mi_piv if c in mi_piv[k]}).dropna(subset=["close"])
        # adjustment factor per date = yahoo adj close / twse close (1.0 on latest)
        fac = (y.reindex(tw.index) / tw.close).where(lambda s: (s > 0.02) & (s < 1.05))  # incl. splits
        fac = fac.fillna(1.0)
        if len(fac):
            fac.iloc[-1] = 1.0
        adj = tw[["open", "high", "low", "close"]].mul(fac, axis=0)
        # long close series: yahoo before first TWSE day, adjusted TWSE after
        lc = pd.concat([y[y.index < tw.index[0]], adj.close]).sort_index()
        lc = lc[~lc.index.duplicated(keep="last")]
        close = adj.close.iloc[-1]
        prev_c = adj.close.shift(1)
        tr = pd.concat([adj.high - adj.low, (adj.high - prev_c).abs(), (adj.low - prev_c).abs()], axis=1).max(axis=1)
        atr = tr.iloc[-14:].mean()  # simple 14-day ATR
        n = len(lc)
        rec = dict(code=c, name=liquid.at[c, "name"], kind=liquid.at[c, "kind"], close=close,
                   turn20_m=liquid.at[c, "turn20"] / 1e6, hist_days=n,
                   ma10=adj.close.iloc[-10:].mean(), ma20=adj.close.iloc[-20:].mean(),
                   ma60=lc.iloc[-60:].mean() if n >= 60 else np.nan,
                   ret20=close / lc.iloc[-21] - 1 if n >= 21 else np.nan,
                   ret60=close / lc.iloc[-61] - 1 if n >= 61 else np.nan,
                   atr=atr, atr_pct=atr / close,
                   high20=adj.high.iloc[-20:].max(), low20=adj.low.iloc[-20:].min(),
                   high60=max(adj.high.iloc[-20:].max(), lc.iloc[-60:].max()),
                   sell_ret=close / lc.loc[SELLOFF_BASE] - 1 if SELLOFF_BASE in lc.index else np.nan,
                   yahoo_close_dev=(y.iloc[-1] / close - 1) if len(y) and y.index[-1] == TARGET else np.nan)
        out.append(rec)
    df = pd.DataFrame(out).set_index("code")
    df = df.join(liquid[["vol20", "excluded", "attention"]])

    # ---- institutional flow ----
    t86 = pd.concat([load_t86(d) for d in t86_days])
    flow = t86.groupby("code")[["foreign", "trust"]].sum()
    df = df.join(flow / 1000.0)  # in lots (張)
    df["inst_net_lots"] = df.foreign.fillna(0) + df.trust.fillna(0)
    df["inst10_pct"] = df.inst_net_lots * 1000 / df.vol20 * 100
    df["foreign10_pct"] = df.foreign.fillna(0) * 1000 / df.vol20 * 100
    df["trust10_pct"] = df.trust.fillna(0) * 1000 / df.vol20 * 100

    df["c_ma20"] = df.close / df.ma20 - 1
    df["c_ma60"] = df.close / df.ma60 - 1
    df["rs20"] = df.ret20 - tw_r20
    df["rs60"] = df.ret60 - tw_r60
    df["dist_high20"] = df.close / df.high20 - 1
    df["sell_rel"] = df.sell_ret - tw_sell

    # ---- composite (ranked within kind; excluded names ranked too but flagged) ----
    def pr(s, asc=True):
        return s.rank(pct=True, ascending=asc).fillna(0.5) * 100

    parts = []
    for kind, g in df.groupby("kind"):
        s = pd.DataFrame(index=g.index)
        s["sc_trend"] = (0.10 * pr(g.rs60) + 0.10 * pr(g.rs20) + 0.075 * pr(g.c_ma20) + 0.075 * pr(g.c_ma60)) / 0.35
        s["sc_flow"] = pr(g.inst10_pct)
        s["sc_resil"] = pr(g.sell_rel)
        s["sc_risk"] = (pr(g.atr_pct, asc=False) + pr(g.dist_high20)) / 2
        pen = np.where(g.c_ma20 > 0.15, -10, 0) + np.where(g.atr_pct > 0.06, -10, 0) \
            + np.where(g.attention.str.contains("注意"), -5, 0)
        s["penalty"] = pen
        s["score"] = 0.35 * s.sc_trend + 0.25 * s.sc_flow + 0.20 * s.sc_resil + 0.20 * s.sc_risk + s.penalty
        parts.append(s)
    df = df.join(pd.concat(parts))

    # ---- trade plan ----
    plans = []
    for c, r in df.iterrows():
        etf = r.kind == "etf"
        a = r.atr
        hi = min(r.close, max(r.ma10, r.ma20) + 0.5 * a)
        lo = max(min(r.ma10, r.ma20), r.close - 1.5 * a)
        if lo >= hi:
            lo = hi - 0.5 * a
        mid = (lo + hi) / 2
        stop = mid - 2.0 * a
        stop = min(max(stop, mid * 0.91), mid * 0.94)  # clip to -6%..-9% from mid
        risk = mid - stop
        tgt = mid + 1.5 * risk  # first target = 1.5R, or prior 20d high if it sits between 1R and 1.5R
        if mid + 1.0 * risk <= r.high20 < tgt:
            tgt = r.high20
        plans.append(dict(code=c, entry_lo=rnd(lo, etf, "down"), entry_hi=rnd(hi, etf, "up"),
                          stop=rnd(stop, etf, "down"), target1=rnd(tgt, etf, "up"),
                          stop_pct=stop / mid - 1, tgt_pct=tgt / mid - 1, stop_atr=(mid - stop) / a))
    df = df.join(pd.DataFrame(plans).set_index("code"))
    # industry + auto notes
    IND = {"01": "水泥", "02": "食品", "03": "塑膠", "04": "紡織", "05": "電機機械", "06": "電器電纜", "08": "玻璃陶瓷",
           "09": "造紙", "10": "鋼鐵", "11": "橡膠", "12": "汽車", "14": "建材營造", "15": "航運", "16": "觀光餐旅",
           "17": "金融", "18": "貿易百貨", "19": "綜合", "20": "其他", "21": "化學", "22": "生技醫療", "23": "油電燃氣",
           "24": "半導體", "25": "電腦週邊", "26": "光電", "27": "通信網路", "28": "電子零組件", "29": "電子通路",
           "30": "資訊服務", "31": "其他電子", "35": "綠能環保", "36": "數位雲端", "37": "運動休閒", "38": "居家生活"}
    try:
        cb = {r["公司代號"].strip(): IND.get(r["產業別"].strip(), r["產業別"].strip())
              for r in json.load(open(os.path.join(RAW, "company_basic.json")))}
    except Exception:
        cb = {}
    df["industry"] = [cb.get(c, "ETF" if df.at[c, "kind"].startswith("etf") else "") for c in df.index]

    def note(r):
        n = []
        if r.stop_atr < 1.5:
            n.append(f"ATR高(停損僅{r.stop_atr:.1f}ATR)宜減碼")
        if r.entry_hi < r.close:
            n.append("等拉回")
        if r.c_ma20 > 0.08:
            n.append("離月線遠")
        if r.inst10_pct < 0:
            n.append("法人10日賣超")
        if r.close < r.ma60:
            n.append("低於季線")
        if r.attention:
            n.append(r.attention)
        return "；".join(n)
    df["note"] = df.apply(note, axis=1)
    df["rank_in_kind"] = df.groupby("kind")["score"].rank(ascending=False, method="first")
    eligible = (df.excluded == "").values
    df["eligible"] = eligible
    df = df.sort_values(["kind", "score"], ascending=[False, False])
    rcols = ["ret20", "ret60", "rs20", "rs60", "c_ma20", "c_ma60", "atr_pct", "dist_high20", "sell_ret", "sell_rel",
             "stop_pct", "tgt_pct", "yahoo_close_dev"]
    outp = df.copy()
    for k in rcols:
        outp[k] = (outp[k] * 100).round(2)
    for k in ["stop_atr", "ma10", "ma20", "ma60", "atr", "high20", "low20", "high60", "turn20_m", "inst10_pct", "foreign10_pct",
              "trust10_pct", "foreign", "trust", "inst_net_lots", "score", "sc_trend", "sc_flow", "sc_resil", "sc_risk"]:
        outp[k] = outp[k].round(2)
    outp.drop(columns=["vol20"]).to_csv(os.path.join(HERE, f"screen_{TARGET}.csv"), encoding="utf-8-sig")
    meta = dict(target=TARGET, mi_days=mi_days, t86_days=t86_days, taiex=twii.iloc[-1], tw_r20=tw_r20, tw_r60=tw_r60,
                tw_sell=tw_sell, sell_base=SELLOFF_BASE, n_liquid=len(df), n_stock=int((df.kind == "stock").sum()),
                n_etf=int((df.kind == "etf").sum()), excluded=df[df.excluded != ""].excluded.to_dict(), gaps=gaps,
                short_hist=df[df.hist_days < 61].index.tolist())
    json.dump(meta, open(os.path.join(HERE, f"meta_{TARGET}.json"), "w"), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(meta, ensure_ascii=False, default=float, indent=1))
    show = ["name", "industry", "close", "ret20", "rs60", "inst10_pct", "atr_pct", "dist_high20", "sell_rel", "c_ma20", "score",
            "entry_lo", "entry_hi", "stop", "target1", "attention"]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    show += ["stop_atr", "turn20_m", "c_ma60", "rs20", "note"]
    el = outp.excluded == ""
    print(outp[(outp.kind == "stock") & el][show].head(30).to_string())
    print(outp[(outp.kind == "etf") & el][show].head(12).to_string())
    print(outp[(outp.kind == "etf_bond") & el][show].to_string())


if __name__ == "__main__":
    main()
