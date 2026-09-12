# =====================================================================
# diagnostic_v14_kfold.py
# =====================================================================
# Purpose: K-Fold consistency validation
# Config:  4h timeframe, MTF on, kNN on, trailing disabled
# Method:  Split data into N windows, run each independently,
#          compute consistency score
# Date:    2026-09-11
# =====================================================================

import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
import pandas as pd
import math
from scipy import stats
import yfinance as yf


# =====================================================================
# CONFIGURATION
# =====================================================================
SYMBOL = "BTC-USD"
TIMEFRAME = "4h"
DATA_PERIOD = "730d"
INITIAL_CAPITAL = 100_000.0
POSITION_PCT = 0.10
COMMISSION = 0.0005
SLIPPAGE = 0.0002

PATTERN_LEN = 10
MEMORY_SIZE = 80
K_NEIGHBORS = 5
AHEAD = 2
PRED_SMOOTH = 4
MIN_PRED_ATR = 0.20

USE_MTF = True
HTF_TIMEFRAME = "4h"
HTF_EMA_LEN = 50

USE_VOLATILITY = True
ATR_LEN = 14
MIN_ATR_PCT = 0.25
BB_LEN = 20
BB_MULT = 2.0
MIN_BB_WIDTH_PCT = 1.0
USE_VOLUME = True
VOLUME_LEN = 20
VOLUME_MULT = 1.1
USE_ADX = True
ADX_LEN = 14
MIN_ADX = 18.0

STOP_ATR = 2.0
TP_ATR = 3.5
TRAIL_ATR = 999.0

# K-Fold
N_FOLDS = 6


# =====================================================================
# DATA LOADING
# =====================================================================
def wilder(s, p):
    return s.ewm(alpha=1.0 / p, adjust=False, min_periods=p).mean()


def add_indicators(df):
    prev = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"],
                    (df["High"] - prev).abs(),
                    (df["Low"] - prev).abs()], axis=1).max(axis=1)
    df["ATR"] = wilder(tr, ATR_LEN)
    df["ATR_PCT"] = np.where(df["Close"] > 0,
                              df["ATR"] / df["Close"] * 100.0, np.nan)

    df["EMA_FAST"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["EMA_SLOW"] = df["Close"].ewm(span=30, adjust=False).mean()

    d = df["Close"].diff()
    g = d.clip(lower=0.0)
    l = -d.clip(upper=0.0)
    ag = wilder(g, 14)
    al = wilder(l, 14)
    rs = ag / al.replace(0.0, np.nan)
    df["RSI"] = (100.0 - 100.0 / (1.0 + rs)).fillna(50.0)

    bb_b = df["Close"].rolling(BB_LEN).mean()
    bb_s = df["Close"].rolling(BB_LEN).std(ddof=1)
    df["BB_WIDTH_PCT"] = np.where(bb_b > 0,
        (2.0 * BB_MULT * bb_s / bb_b) * 100.0, np.nan)

    df["VOL_SMA"] = df["Volume"].rolling(VOLUME_LEN).mean()
    df["VOL_CONF"] = (df["Volume"].notna() & df["VOL_SMA"].notna()
                      & (df["Volume"] > df["VOL_SMA"] * VOLUME_MULT))

    up = df["High"].diff()
    dn = -df["Low"].diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    ps = wilder(pdm, ADX_LEN)
    ms = wilder(mdm, ADX_LEN)
    as_ = df["ATR"].replace(0.0, np.nan)
    dip = 100.0 * ps / as_
    dim = 100.0 * ms / as_
    dis = (dip + dim).replace(0.0, np.nan)
    dx = 100.0 * (dip - dim).abs() / dis
    df["ADX"] = wilder(dx.fillna(0.0), ADX_LEN)

    htf_c = df["Close"].resample(HTF_TIMEFRAME).last()
    htf_e = htf_c.ewm(span=HTF_EMA_LEN, adjust=False).mean()
    df["HTF_EMA"] = htf_e.shift(1).reindex(df.index, method="ffill")
    df["HTF_BULL"] = df["Close"] > df["HTF_EMA"]
    df["HTF_BEAR"] = df["Close"] < df["HTF_EMA"]

    vn = np.maximum(df["ATR"] / df["Close"], 1e-6)
    f1 = np.log(df["Close"] / df["Close"].shift(1)) / vn
    f2 = (df["Close"] - df["Close"].shift(5)) / df["ATR"]
    f3 = (df["RSI"] - 50.0) / 50.0
    f4 = (df["EMA_FAST"] - df["EMA_SLOW"]) / df["ATR"]
    for c, s in [("f1", f1), ("f2", f2), ("f3", f3), ("f4", f4)]:
        df[c] = np.clip(s.replace([np.inf, -np.inf], 0.0).fillna(0.0),
                        -5.0, 5.0)
    return df


def load_data():
    print(f"Loading {SYMBOL} {TIMEFRAME}...")
    df = yf.download(SYMBOL, interval=TIMEFRAME, period=DATA_PERIOD,
                     auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
    print(f"Total bars: {len(df)}")
    df = add_indicators(df)
    return df


# =====================================================================
# BLOCK LENGTH ESTIMATION
# =====================================================================
def estimate_block_length(preds, max_lag=100, z=1.96):
    s = np.asarray(preds, dtype=float)
    s = s[np.isfinite(s)]
    if len(s) < max_lag * 2:
        return max(AHEAD, PATTERN_LEN + PRED_SMOOTH)
    s = s - s.mean()
    if s.var() < 1e-12:
        return max(AHEAD, PATTERN_LEN + PRED_SMOOTH)
    n = len(s)
    acf = np.correlate(s, s, mode='full')[n - 1:] / (s.var() * np.arange(n, 0, -1))
    thr = z / np.sqrt(n)
    for lag in range(1, max_lag):
        if abs(acf[lag]) < thr:
            return max(lag * 2, AHEAD)
    return max_lag


# =====================================================================
# BACKTEST ENGINE
# =====================================================================
def run_engine(df, comm, slip):
    o = df["Open"].values.astype(float)
    h = df["High"].values.astype(float)
    lo = df["Low"].values.astype(float)
    c = df["Close"].values.astype(float)
    a = df["ATR"].values.astype(float)
    ap = df["ATR_PCT"].values.astype(float)
    f1 = df["f1"].values.astype(float)
    f2 = df["f2"].values.astype(float)
    f3 = df["f3"].values.astype(float)
    f4 = df["f4"].values.astype(float)
    hb = df["HTF_BULL"].fillna(False).astype(bool).values
    hbe = df["HTF_BEAR"].fillna(False).astype(bool).values
    bw = df["BB_WIDTH_PCT"].values.astype(float)
    ax = df["ADX"].values.astype(float)
    vc = df["VOL_CONF"].fillna(False).astype(bool).values

    n = len(df)
    X = np.zeros((MEMORY_SIZE, PATTERN_LEN * 4))
    Y = np.zeros(MEMORY_SIZE)
    head = 0
    count = 0
    pred_ema = None
    alpha = 2.0 / (PRED_SMOOTH + 1.0) if PRED_SMOOTH > 1 else 1.0

    equity = INITIAL_CAPITAL
    pos = 0
    qty = 0.0
    ep = 0.0
    eb = 0.0
    eb_i = None
    ecm = 0.0
    istop = 0.0
    tstop = 0.0
    tgt = 0.0
    pend = 0
    patr = np.nan

    curve = np.zeros(n)
    trades = []
    bars = 0
    pre = []

    req = max(350, PATTERN_LEN + AHEAD + 50, HTF_EMA_LEN * 4 + 50)

    def do_exit(i, xp, reason):
        nonlocal equity, pos, qty, ep, eb, eb_i, ecm, istop, tstop, tgt
        if pos == 0 or qty <= 0:
            return

        if pos == 1:
            fill = xp * (1.0 - slip)
            gr = (fill - ep) * qty
        else:
            fill = xp * (1.0 + slip)
            gr = (ep - fill) * qty
        xc = fill * qty * comm
        equity += gr - xc

        ig = (xp - eb) * qty if pos == 1 else (eb - xp) * qty
        es = abs(ep - eb) * qty
        xs = abs(fill - xp) * qty
        tc = es + xs + ecm + xc
        net = ig - tc

        trades.append({
            "signal_bar": (eb_i - 1) if eb_i is not None else "",
            "entry_bar": eb_i if eb_i is not None else "",
            "exit_bar": i,
            "direction": "long" if pos == 1 else "short",
            "ideal_entry": eb,
            "actual_entry": ep,
            "ideal_exit": xp,
            "actual_exit": fill,
            "qty": qty,
            "ideal_gross": ig,
            "entry_slip": es,
            "exit_slip": xs,
            "entry_comm": ecm,
            "exit_comm": xc,
            "total_cost": tc,
            "net_pnl": net,
            "exit_reason": reason,
        })

        pos = 0
        qty = 0.0
        ep = 0.0
        eb = 0.0
        eb_i = None
        ecm = 0.0
        istop = 0.0
        tstop = 0.0
        tgt = 0.0

    for i in range(n):
        if pend != 0 and i > 0:
            if pos != 0 and pos != pend:
                do_exit(i, o[i], "signal_reverse")
            if pos == 0:
                d = pend
                ea = patr if np.isfinite(patr) else a[i]
                if np.isfinite(ea) and ea > 0 and equity > 0:
                    eb = o[i]
                    ep = eb * (1.0 + slip) if d == 1 else eb * (1.0 - slip)
                    qty = (equity * POSITION_PCT) / ep
                    eb_i = i
                    ecm = ep * qty * comm
                    equity -= ecm
                    if d == 1:
                        istop = ep - STOP_ATR * ea
                        tgt = ep + TP_ATR * ea
                    else:
                        istop = ep + STOP_ATR * ea
                        tgt = ep - TP_ATR * ea
                    tstop = istop
                    pos = d
            pend = 0
            patr = np.nan

        if pos != 0:
            if pos == 1:
                es = max(istop, tstop)
                if o[i] <= es:
                    do_exit(i, o[i], "gap_stop")
                elif lo[i] <= es:
                    do_exit(i, es, "stop")
                elif o[i] >= tgt:
                    do_exit(i, o[i], "gap_tp")
                elif h[i] >= tgt:
                    do_exit(i, tgt, "tp")
            else:
                es = min(istop, tstop)
                if o[i] >= es:
                    do_exit(i, o[i], "gap_stop")
                elif h[i] >= es:
                    do_exit(i, es, "stop")
                elif o[i] <= tgt:
                    do_exit(i, o[i], "gap_tp")
                elif lo[i] <= tgt:
                    do_exit(i, tgt, "tp")

        if pos != 0 and np.isfinite(a[i]) and a[i] > 0:
            if pos == 1:
                tstop = max(tstop, c[i] - TRAIL_ATR * a[i])
            else:
                tstop = min(tstop, c[i] + TRAIL_ATR * a[i])

        if pos != 0:
            bars += 1

        rp = None
        if i >= req:
            te = i - AHEAD
            if te >= PATTERN_LEN - 1:
                s0 = te - PATTERN_LEN + 1
                vec = np.concatenate([f1[s0:te + 1], f2[s0:te + 1],
                                       f3[s0:te + 1], f4[s0:te + 1]])
                if c[i] > 0 and c[te] > 0:
                    lab = math.log(c[i] / c[te])
                    idx = head % MEMORY_SIZE
                    X[idx] = vec
                    Y[idx] = lab
                    head += 1
                    count = min(count + 1, MEMORY_SIZE)

            if i >= PATTERN_LEN - 1:
                s0 = i - PATTERN_LEN + 1
                q = np.concatenate([f1[s0:i + 1], f2[s0:i + 1],
                                     f3[s0:i + 1], f4[s0:i + 1]])
                if count >= K_NEIGHBORS:
                    vX = X[:count] if count < MEMORY_SIZE else X
                    vY = Y[:count] if count < MEMORY_SIZE else Y
                    ds = np.linalg.norm(vX - q, axis=1)
                    if K_NEIGHBORS < len(ds):
                        t = np.argpartition(ds, K_NEIGHBORS - 1)[:K_NEIGHBORS]
                    else:
                        t = np.argsort(ds)
                    dt = ds[t]
                    yt = vY[t]
                    w = 1.0 / (dt + 1e-6)
                    ws = w.sum()
                    if ws > 0:
                        rp = float((w * yt).sum() / ws)

            if rp is not None:
                if pred_ema is None:
                    pred_ema = rp
                else:
                    pred_ema = alpha * rp + (1 - alpha) * pred_ema

        if (pred_ema is not None and np.isfinite(ap[i]) and ap[i] > 0
                and i + AHEAD < n):
            fr = math.log(c[i + AHEAD] / c[i]) if c[i] > 0 else 0.0
            pre.append({
                "bar": i,
                "pred": float(pred_ema),
                "score": float(pred_ema / (ap[i] / 100.0)),
                "fwd_ret": float(fr),
            })

        if (i < n - 1 and pred_ema is not None
                and np.isfinite(ap[i]) and ap[i] > 0):
            sc = pred_ema / (ap[i] / 100.0)
            vr = (np.isfinite(ap[i]) and np.isfinite(bw[i])
                  and ap[i] >= MIN_ATR_PCT and bw[i] >= MIN_BB_WIDTH_PCT)
            ar = np.isfinite(ax[i]) and ax[i] >= MIN_ADX
            lf = ((not USE_MTF or hb[i])
                  and (not USE_VOLATILITY or vr)
                  and (not USE_VOLUME or vc[i])
                  and (not USE_ADX or ar))
            sf = ((not USE_MTF or hbe[i])
                  and (not USE_VOLATILITY or vr)
                  and (not USE_VOLUME or vc[i])
                  and (not USE_ADX or ar))
            if sc >= MIN_PRED_ATR and lf and pos <= 0:
                pend = 1
                patr = a[i]
            elif sc <= -MIN_PRED_ATR and sf and pos >= 0:
                pend = -1
                patr = a[i]

        unr = 0.0
        if pos == 1:
            unr = (c[i] - ep) * qty
        elif pos == -1:
            unr = (ep - c[i]) * qty
        curve[i] = equity + unr

    if pos != 0:
        do_exit(n - 1, c[-1], "end_of_data")
        curve[-1] = equity

    return curve, trades, bars, pre


# =====================================================================
# SUMMARY HELPER
# =====================================================================
def summarize(trades, initial_capital=INITIAL_CAPITAL):
    if not trades:
        return {"trades": 0, "gross": 0.0, "net": 0.0,
                "return_pct": 0.0, "wr": 0.0, "pf": 0.0}
    t = pd.DataFrame(trades)
    ig = t["ideal_gross"].sum()
    net = t["net_pnl"].sum()
    wins = t[t["net_pnl"] > 0]["net_pnl"]
    losses = t[t["net_pnl"] <= 0]["net_pnl"]
    wr = 100.0 * len(wins) / len(t) if len(t) > 0 else 0.0
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else float("nan")
    return {
        "trades": len(t),
        "gross": ig,
        "net": net,
        "return_pct": net / initial_capital * 100.0,
        "wr": wr,
        "pf": pf,
    }


# =====================================================================
# MAIN - K-Fold Validation
# =====================================================================
def main():
    df = load_data()
    n = len(df)

    # Build fold boundaries
    fold_size = n // N_FOLDS
    boundaries = []
    for k in range(N_FOLDS):
        start = k * fold_size
        end = (k + 1) * fold_size if k < N_FOLDS - 1 else n
        boundaries.append((start, end))

    print(f"\n{N_FOLDS}-Fold Windows:")
    for k, (s, e) in enumerate(boundaries):
        print(f"  Fold {k+1}: bars [{s:>5} : {e:>5}]  "
              f"{df.index[s].date()} -> {df.index[e-1].date()}  "
              f"({e - s} bars)")

    # Run each fold
    print("\n" + "=" * 100)
    print(f"{'Fold':<6} {'Period':<25} {'Trd':>5} {'Gross':>12} {'Net':>12} "
          f"{'Ret%':>8} {'WR%':>8} {'PF':>8} {'IC':>9}")
    print("-" * 100)

    fold_results = []
    positive_count = 0
    negative_count = 0

    for k, (s, e) in enumerate(boundaries):
        df_fold = df.iloc[s:e].copy()
        curve, trades, bars, pre = run_engine(df_fold,
                                                COMMISSION, SLIPPAGE)
        sm = summarize(trades)

        # IC on this fold
        if len(pre) > 50:
            p = np.array([r["pred"] for r in pre], dtype=float)
            f = np.array([r["fwd_ret"] for r in pre], dtype=float)
            if np.std(p) > 1e-9:
                ic = stats.spearmanr(p, f)[0]
            else:
                ic = float("nan")
        else:
            ic = float("nan")

        period = f"{df.index[s].date()} -> {df.index[e-1].date()}"
        print(f"{k+1:<6} {period:<25} {sm['trades']:>5} "
              f"${sm['gross']:>11,.0f} ${sm['net']:>11,.0f} "
              f"{sm['return_pct']:>7.2f}% {sm['wr']:>7.2f} "
              f"{sm['pf']:>8.3f} {ic:>+9.5f}")

        if sm["net"] > 0:
            positive_count += 1
        else:
            negative_count += 1

        fold_results.append({
            "fold": k + 1,
            "period": period,
            "net": sm["net"],
            "gross": sm["gross"],
            "trades": sm["trades"],
            "wr": sm["wr"],
            "pf": sm["pf"],
            "ic": ic,
        })

    print("=" * 100)

    # Consistency analysis
    total_folds = N_FOLDS
    pos_ratio = positive_count / total_folds

    print(f"\n{'=' * 70}")
    print("CONSISTENCY ANALYSIS")
    print(f"{'=' * 70}")
    print(f"  Positive folds: {positive_count} / {total_folds} "
          f"({pos_ratio * 100:.1f}%)")
    print(f"  Negative folds: {negative_count} / {total_folds} "
          f"({(1 - pos_ratio) * 100:.1f}%)")

    # Sum and mean
    total_net = sum(r["net"] for r in fold_results)
    mean_net = total_net / total_folds
    std_net = np.std([r["net"] for r in fold_results], ddof=1) \
        if total_folds > 1 else 0.0
    mean_ret = sum(r["net"] for r in fold_results) / INITIAL_CAPITAL * 100.0

    print(f"  Total Net (all folds): ${total_net:,.2f}")
    print(f"  Mean Net per fold:     ${mean_net:,.2f}")
    print(f"  Std Net per fold:      ${std_net:,.2f}")
    print(f"  Total Return %:        {mean_ret:+.2f}%")

    # Verdict
    print(f"\n{'=' * 70}")
    print("VERDICT")
    print(f"{'=' * 70}")

    if pos_ratio >= 0.70:
        print(f"[STRONG] {positive_count}/{total_folds} folds positive "
              f"({pos_ratio * 100:.0f}%)")
        print("         Edge appears robust across regimes.")
        print("         Next step: parameter stability + live testing.")
    elif pos_ratio >= 0.50:
        print(f"[MODERATE] {positive_count}/{total_folds} folds positive "
              f"({pos_ratio * 100:.0f}%)")
        print("           Edge is regime-dependent.")
        print("           Next step: identify which regimes are profitable.")
    elif pos_ratio >= 0.33:
        print(f"[WEAK] {positive_count}/{total_folds} folds positive "
              f"({pos_ratio * 100:.0f}%)")
        print("       Most folds lose money.")
        print("       Next step: feature space redesign.")
    else:
        print(f"[FAIL] {positive_count}/{total_folds} folds positive "
              f"({pos_ratio * 100:.0f}%)")
        print("       No consistent edge.")
        print("       Next step: stop or abandon this approach.")

    # Regime context: BTC price change per fold
    print(f"\n{'=' * 70}")
    print("REGIME CONTEXT (BTC price change per fold)")
    print(f"{'=' * 70}")
    for k, (s, e) in enumerate(boundaries):
        p_start = df["Close"].iloc[s]
        p_end = df["Close"].iloc[e - 1]
        chg = (p_end / p_start - 1.0) * 100.0
        fold_net = fold_results[k]["net"]
        outcome = "WIN " if fold_net > 0 else "LOSS"
        print(f"  Fold {k+1}: BTC {chg:+7.2f}%   |   strategy "
              f"${fold_net:+10,.0f}   [{outcome}]")


if __name__ == "__main__":
    main()