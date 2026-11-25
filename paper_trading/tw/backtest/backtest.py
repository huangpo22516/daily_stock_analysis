#!/usr/bin/env python3
"""台股法人籌碼/季節性/營收假說回測（樣本內/樣本外）。

前置：python fetch_prices.py ; python fetch_finmind.py inst revenue
執行：python backtest.py            -> 產出 output/tables.md 與 output/*.csv

方法摘要（細節見 RESULTS.md）：
- 訊號日 t 收盤後才知道法人/營收資料；t+1 開盤進場（還原開盤價），
  t+1 無量（停牌/資料缺）或開盤即漲停（>=+9.5%）視為買不到，剔除。
- h 日報酬 = C[t+h] / O[t+1] - 1；大盤同窗口 ^TWII。
- 超額(對大盤) = 個股 - TAIEX；扣來回成本 0.585%。
- 超額(對宇宙) = 個股 - 同日全宇宙平均（去除存活者偏差的「共同漂移」與股利差異），
  作為顯著性檢定的主要指標。
- t 值：以「訊號日」叢集與以「月份」叢集（較保守，處理持有期重疊）。
"""
import glob
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "output")
COST = 0.00585
IS_END = pd.Timestamp("2021-12-31")
OOS_START = pd.Timestamp("2022-01-01")
START = pd.Timestamp("2015-01-01")
HORIZONS = (5, 10, 20)
COOLDOWN = 20
MKT = "^TWII"


# ----------------------------------------------------------------- data
def load_prices():
    p = os.path.join(DATA, "prices")
    rd = lambda k: pd.read_csv(os.path.join(p, f"{k}.csv.gz"), index_col=0, parse_dates=True)
    C, O, V = rd("close"), rd("open"), rd("volume")
    cal = C.index[C.notna().sum(axis=1) > 100]
    C, O, V = C.loc[cal], O.loc[cal], V.loc[cal]
    C[MKT] = C[MKT].ffill()
    O[MKT] = O[MKT].fillna(C[MKT].shift())
    stocks = [c for c in C.columns if c != MKT]
    tradable = (V[stocks] > 0) & O[stocks].notna() & C[stocks].notna()
    return C, O, V, stocks, tradable


def load_inst(index, stocks):
    d = os.path.join(DATA, "finmind", "TaiwanStockInstitutionalInvestorsBuySell")
    trust, foreign = {}, {}
    for f in sorted(glob.glob(os.path.join(d, "*.csv.gz"))):
        code = os.path.basename(f).split(".")[0]
        if code not in stocks:
            continue
        df = pd.read_csv(f)
        if df.empty:
            continue
        df["net"] = df["buy"] - df["sell"]
        pv = df.pivot_table(index="date", columns="name", values="net", aggfunc="sum")
        pv.index = pd.to_datetime(pv.index)
        trust[code] = pv.get("Investment_Trust")
        foreign[code] = pv.get("Foreign_Investor", 0) + pv.get("Foreign_Dealer_Self", pd.Series(0, index=pv.index)).fillna(0)
    T = pd.DataFrame(trust).reindex(index)
    F = pd.DataFrame(foreign).reindex(index)
    return T, F


def load_revenue(stocks):
    d = os.path.join(DATA, "finmind", "TaiwanStockMonthRevenue")
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "*.csv.gz"))):
        code = os.path.basename(f).split(".")[0]
        if code not in stocks:
            continue
        try:
            df = pd.read_csv(f)
        except pd.errors.EmptyDataError:
            continue
        if df.empty:
            continue
        df["code"] = code
        rows.append(df[["code", "revenue_year", "revenue_month", "revenue"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ----------------------------------------------------------------- returns
def forward_returns(C, O, stocks, tradable):
    """dict h -> (X: 超額對大盤(未扣成本), A: 超額對宇宙平均), 以訊號日 t 為索引。"""
    o1 = O.shift(-1)
    c0 = C
    fill_ok = tradable.shift(-1, fill_value=False) & ((o1[stocks] / c0[stocks] - 1) < 0.095)
    Cf = C[stocks].ffill(limit=5)
    res = {}
    for h in HORIZONS:
        r = Cf.shift(-h) / o1[stocks] - 1
        r = r.where(fill_ok)
        r = r.where(r.abs() < 3)  # 防呆：剔除資料異常
        rm = C[MKT].shift(-h) / o1[MKT] - 1
        X = r.sub(rm, axis=0)
        A = r.sub(r.mean(axis=1), axis=0)
        res[h] = (X, A)
    return res


def regime(C):
    ma = C[MKT].rolling(60).mean()
    return pd.Series(np.where(C[MKT] < ma, "bear(<MA60)", "bull(>=MA60)"), index=C.index)


# ----------------------------------------------------------------- stats
def cl_t(x, groups):
    x = np.asarray(x, float)
    n = len(x)
    if n < 3:
        return np.nan
    g = pd.Series(x - x.mean()).groupby(np.asarray(groups)).sum()
    G = len(g)
    if G < 2:
        return np.nan
    var = (g ** 2).sum() * G / (G - 1) / n ** 2
    return x.mean() / np.sqrt(var) if var > 0 else np.nan


def summarize(df, h):
    """df: 列=訊號，欄含 date, x{h}, a{h}。回傳統計 dict（%）。"""
    x = df[f"x{h}"].dropna()
    sub = df.loc[x.index]
    if len(x) == 0:
        return dict(n=0)
    xn = x - COST
    a = sub[f"a{h}"]
    month = sub["date"].dt.to_period("M").astype(str)
    return dict(
        n=len(x),
        mean_net=100 * xn.mean(),
        median_net=100 * xn.median(),
        hit=100 * (xn > 0).mean(),
        t_date=cl_t(xn, sub["date"]),
        t_month=cl_t(xn, month),
        mean_vs_univ=100 * a.mean(),
        t_univ_month=cl_t(a, month),
    )


def attach(sig, fr, reg):
    """sig: DataFrame(date, code) -> 加上 x/a 各 horizon 與 regime/year/period。"""
    sig = sig.copy()
    for h, (X, A) in fr.items():
        sx, sa = X.stack(), A.stack()
        idx = pd.MultiIndex.from_arrays([sig["date"], sig["code"]])
        sig[f"x{h}"] = sx.reindex(idx).values
        sig[f"a{h}"] = sa.reindex(idx).values
    sig["regime"] = reg.reindex(sig["date"]).values
    sig["year"] = sig["date"].dt.year
    sig["period"] = np.where(sig["date"] <= IS_END, "IS", "OOS")
    return sig


def baseline(fr, reg, idx):
    """全宇宙 stock-day 無條件基準。"""
    out = []
    for h, (X, A) in fr.items():
        X = X.loc[idx]
        st = X.stack().rename("x").reset_index()
        st.columns = ["date", "code", "x"]
        st["period"] = np.where(st["date"] <= IS_END, "IS", "OOS")
        st["regime"] = reg.reindex(st["date"]).values
        for keys, g in [(("ALL",), st)] + [((p,), g) for p, g in st.groupby("period")] + \
                [((p, r), g) for (p, r), g in st.groupby(["period", "regime"])]:
            xn = g["x"] - COST
            out.append(dict(h=h, slice="/".join(keys), n=len(xn), mean_net=100 * xn.mean(),
                            median_net=100 * xn.median(), hit=100 * (xn > 0).mean()))
    return pd.DataFrame(out)


def take_with_cooldown(cond, cooldown=COOLDOWN):
    """cond: bool DataFrame -> DataFrame(date, code)，同檔訊號間隔 >= cooldown 交易日。"""
    rows = []
    idx = cond.index
    for code in cond.columns:
        pos = np.flatnonzero(cond[code].values)
        last = -10 ** 9
        for p in pos:
            if p - last >= cooldown:
                rows.append((idx[p], code))
                last = p
    return pd.DataFrame(rows, columns=["date", "code"])


def run_len_pos(T):
    """連續 trust>0 天數（NaN 中斷）。"""
    pos = (T > 0).astype(int).values
    out = np.zeros_like(pos)
    for i in range(len(pos)):
        out[i] = (out[i - 1] + 1) * pos[i] if i else pos[i]
    return pd.DataFrame(out, index=T.index, columns=T.columns)


def stats_table(sig, horizons, label_cols):
    rows = []
    for keys, g in sig.groupby(label_cols):
        keys = keys if isinstance(keys, tuple) else (keys,)
        for h in horizons:
            s = summarize(g, h)
            rows.append(dict(zip(label_cols, keys), h=h, **s))
    return pd.DataFrame(rows)


def fmt(df, floatfmt="{:.2f}"):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else floatfmt.format(v))
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


# ----------------------------------------------------------------- hypotheses
def eval_event_hyp(name, combos, fr, reg, horizons, primary_h, md):
    """combos: dict label -> bool cond DataFrame。IS 選參（主 horizon 對宇宙超額之月叢集 t 值最大）。"""
    allsig = {}
    grid = []
    for lab, cond in combos.items():
        cond = cond.loc[START:]
        sig = attach(take_with_cooldown(cond.fillna(False)), fr, reg)
        allsig[lab] = sig
        for per in ("IS", "OOS"):
            g = sig[sig.period == per]
            for h in horizons:
                grid.append(dict(params=lab, period=per, h=h, **summarize(g, h)))
    grid = pd.DataFrame(grid)
    is_p = grid[(grid.period == "IS") & (grid.h == primary_h)].set_index("params")["t_univ_month"]
    best = is_p.idxmax()
    sig = allsig[best]
    md.append(f"\n## {name}\n")
    md.append(f"嘗試參數組合 {len(combos)} 組 × horizon {len(horizons)} 個 = {len(combos) * len(horizons)} 個檢定；"
              f"選參規則：樣本內 {primary_h} 日『對宇宙超額』月叢集 t 值最大 → **{best}**\n")
    md.append("### 全部參數網格（IS 與 OOS 並列，mean/median/hit 為扣成本後對大盤超額，%）\n")
    md.append(fmt(grid))
    md.append(f"\n### 選定參數 {best}：期間 × 市場狀態\n")
    md.append(fmt(stats_table(sig, horizons, ["period", "regime"])))
    md.append(f"\n### 選定參數 {best}：逐年（{primary_h} 日）\n")
    md.append(fmt(stats_table(sig, [primary_h], ["year"])))
    sig.to_csv(os.path.join(OUT, f"signals_{name.split()[0]}.csv"), index=False)
    return best, grid, sig


def h2_year_end(C, O, V, T, stocks, tradable, reg, md):
    adv20 = V[stocks].rolling(20).mean()
    t20 = T.rolling(20, min_periods=15).sum()
    score = t20 / adv20 * 100
    idx = C.index
    rows, per_year = [], []
    cur_top = None
    for y in range(2015, 2027):
        nov = idx[(idx.year == y) & (idx.month == 11)]
        dec = idx[(idx.year == y) & (idx.month == 12)]
        if len(nov) == 0:
            continue
        t = nov[-1]
        oct_start = idx[(idx.year == y) & (idx.month == 9)][-1]
        mkt_octnov = C[MKT].loc[t] / C[MKT].loc[oct_start] - 1
        if len(dec) < 15:
            per_year.append(dict(year=y, formation=t.date(), note="12 月尚未結束，無法評估",
                                 mkt_OctNov=100 * mkt_octnov))
            s = score.loc[t, stocks].dropna()
            top = s[s.rank(pct=True) > 0.9].sort_values(ascending=False)
            names = pd.read_csv(os.path.join(HERE, "universe.csv"), dtype={"code": str}).set_index("code")["name"]
            cur_top = pd.DataFrame({"code": top.index, "name": names.reindex(top.index).values,
                                    "trust20_pct_adv": top.values})
            continue
        e0, e1 = dec[0], dec[-1]
        s = score.loc[t, stocks].dropna()
        ok = tradable.loc[e0, s.index] & ((O.loc[e0, s.index] / C.loc[t, s.index] - 1) < 0.095)
        s = s[ok]
        r = (C.loc[e1, s.index].astype(float) / O.loc[e0, s.index] - 1)
        rm = C[MKT].loc[e1] / O[MKT].loc[e0] - 1
        x = r - rm
        q = s.rank(pct=True)
        for grp, mask in [("top10%", q > 0.9), ("top20%", q > 0.8), ("bottom20%", q <= 0.2),
                          ("bottom10%", q <= 0.1), ("universe", q > -1)]:
            rows.append(dict(year=y, group=grp, n=int(mask.sum()), ex_net=100 * (x[mask].mean() - COST),
                             hit=100 * (x[mask] - COST > 0).mean()))
        per_year.append(dict(year=y, formation=t.date(), mkt_OctNov=100 * mkt_octnov,
                             mkt_Dec=100 * rm, n=len(s), bear=reg.loc[t].startswith("bear")))
    R = pd.DataFrame(rows)
    PY = pd.DataFrame(per_year)
    wide = R.pivot(index="year", columns="group", values="ex_net")
    wide["top10-univ"] = wide["top10%"] - wide["universe"]
    wide["top20-univ"] = wide["top20%"] - wide["universe"]
    wide["top20-bot20"] = wide["top20%"] - wide["bottom20%"]
    wide = wide.join(PY.set_index("year")[["mkt_OctNov", "mkt_Dec", "bear"]])
    wide["period"] = np.where(wide.index <= 2021, "IS", "OOS")
    wide["OctNov_down"] = wide["mkt_OctNov"] < 0
    md.append("\n## H2 年底作帳（11 月最後交易日依投信 20 日買超/均量 排名，持有至 12 月最後交易日）\n")
    md.append("參數組合：top10% 與 top20% 兩組（另列 bottom 與全宇宙平均做對照）。數值為扣成本後對大盤超額 %。\n")
    md.append(fmt(wide.reset_index()[["year", "period", "mkt_OctNov", "bear", "mkt_Dec", "top10%", "top20%", "universe",
                                      "bottom20%", "top10-univ", "top20-univ", "top20-bot20"]]))
    summ = []
    for lab, g in [("IS 2015-2021", wide[wide.period == "IS"]), ("OOS 2022-2025", wide[wide.period == "OOS"]),
                   ("ALL", wide), ("Oct-Nov 大盤下跌年", wide[wide.OctNov_down]),
                   ("Oct-Nov 大盤上漲年", wide[~wide.OctNov_down]),
                   ("形成日大盤<MA60", wide[wide.bear == True]), ("形成日大盤>=MA60", wide[wide.bear == False])]:
        d = dict(slice=lab, years=len(g))
        for c in ["top10-univ", "top20-univ", "top20-bot20"]:
            v = g[c].dropna()
            d[f"{c}_mean"] = v.mean()
            d[f"{c}_pos_yrs"] = f"{int((v > 0).sum())}/{len(v)}"
            d[f"{c}_t"] = v.mean() / (v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 2 else np.nan
        summ.append(d)
    md.append("\n### 彙總（以年為單位，t 值 = 年度平均/標準誤）\n")
    md.append(fmt(pd.DataFrame(summ)))
    md.append("\n2026 年形成期資訊：\n")
    md.append(fmt(PY[PY.year == 2026]))
    if cur_top is not None:
        md.append("\n截至最新資料日之投信 20 日買超/均量 前 10%（僅供參考，非回測結果）：\n")
        md.append(fmt(cur_top))
    R.to_csv(os.path.join(OUT, "h2_groups.csv"), index=False)
    return wide


def h4_revenue(rev, C, stocks, fr, reg, md, known_day=10):
    if rev.empty:
        md.append("\n## H4 月營收加速\n\n月營收資料尚未抓取完成，略過。\n")
        return None
    rev = rev.drop_duplicates(["code", "revenue_year", "revenue_month"]).copy()
    rev["ym"] = rev["revenue_year"] * 12 + rev["revenue_month"] - 1
    piv = rev.pivot(index="ym", columns="code", values="revenue").astype(float)
    full = pd.RangeIndex(piv.index.min(), piv.index.max() + 1)
    piv = piv.reindex(full)
    yoy = piv / piv.shift(12) - 1
    yoy = yoy.where(piv.shift(12) > 0)
    cond = (yoy > 0.2) & (yoy.shift(1) > 0.2) & (yoy > yoy.shift(1))
    idx = C.index
    rows = []
    for ym, r in cond.iterrows():
        y, m = divmod(ym, 12)
        m += 1
        # 已知日：次月 10 日收盤後；t = 次月 10 日（含）之前最後交易日，t+1 開盤進場
        ky, km = (y + 1, 1) if m == 12 else (y, m + 1)
        known = pd.Timestamp(ky, km, known_day)
        pos = idx.searchsorted(known, side="right") - 1
        if pos < 0 or pos >= len(idx) - 1 or idx[pos] < START:
            continue
        t = idx[pos]
        for code in r.index[r.fillna(False).values]:
            rows.append((t, code))
    sig = attach(pd.DataFrame(rows, columns=["date", "code"]), fr, reg)
    if known_day != 10:
        md.append(f"\n### H4 穩健性：已知日改為次月 {known_day} 日（防公告延遲/資料修正的 look-ahead）\n")
        md.append(fmt(stats_table(sig, list(HORIZONS), ["period"])))
        md.append("")
        md.append(fmt(stats_table(sig, [20], ["period", "regime"])))
        return sig
    md.append("\n## H4 月營收加速（YoY 本月 > 上月 且兩者皆 > 20%；次月 10 日視為已知日，下一交易日開盤進場）\n")
    md.append("參數固定（未調參）：1 組 × 20 日 horizon（另列 5/10 日參考）。\n")
    md.append(fmt(stats_table(sig, list(HORIZONS), ["period"])))
    md.append("\n### 期間 × 市場狀態（20 日）\n")
    md.append(fmt(stats_table(sig, [20], ["period", "regime"])))
    md.append("\n### 逐年（20 日）\n")
    md.append(fmt(stats_table(sig, [20], ["year"])))
    sig.to_csv(os.path.join(OUT, "signals_H4.csv"), index=False)
    return sig


def main():
    os.makedirs(OUT, exist_ok=True)
    C, O, V, stocks, tradable = load_prices()
    T, F = load_inst(C.index, stocks)
    have = [c for c in stocks if c in T.columns]
    print(f"prices {C.shape}, inst for {len(have)} stocks")
    fr = forward_returns(C, O, stocks, tradable)
    reg = regime(C)
    md = [f"# 自動產出表格（backtest.py）\n\n宇宙：{len(stocks)} 檔（有法人資料 {len(have)} 檔）；"
          f"價格 {C.index[0].date()}~{C.index[-1].date()}；成本 {COST * 100:.3f}% 來回。\n"]

    # 基準
    valid_idx = C.index[(C.index >= START)]
    base = baseline(fr, reg, valid_idx)
    md.append("\n## 無條件基準（全宇宙所有 stock-day，t+1 開盤進場，扣成本後對大盤超額 %）\n")
    md.append(fmt(base))
    base.to_csv(os.path.join(OUT, "baseline.csv"), index=False)

    Th, Fh = T[have], F[have]
    adv20 = V[have].rolling(20).mean()
    # H1
    rl = run_len_pos(Th)
    combos = {}
    for N in (3, 5):
        cum = Th.rolling(N).sum()
        for X in (5, 10):
            combos[f"N={N},X={X}%"] = (rl >= N) & (cum / adv20 * 100 >= X)
    eval_event_hyp("H1 投信連買", combos, fr, reg, list(HORIZONS), 20, md)

    # H3
    f10 = Fh.rolling(10).sum() / adv20 * 100
    t10 = Th.rolling(10).sum() / adv20 * 100
    combos3 = {"both>0": (f10 > 0) & (t10 > 0), "both>=10%": (f10 >= 10) & (t10 >= 10)}
    eval_event_hyp("H3 外資投信同步買超", combos3, fr, reg, [10, 20], 20, md)

    # H2
    h2_year_end(C, O, V, Th, have, tradable[have], reg, md)

    # H4
    rev = load_revenue(stocks)
    h4_revenue(rev, C, stocks, fr, reg, md)
    if not rev.empty:
        h4_revenue(rev, C, stocks, fr, reg, md, known_day=15)

    with open(os.path.join(OUT, "tables.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("wrote output/tables.md")


if __name__ == "__main__":
    main()
