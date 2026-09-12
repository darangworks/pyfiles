# =====================================================================
# diagnostic_v15_crossasset.py
# =====================================================================
# Purpose: Cross-asset validation of the kNN strategy
# Assets:  BTC-USD, ETH-USD, SOL-USD
# Method:  6-Fold K-Fold validation per asset
# Config:  4h timeframe, MTF on, kNN on, trailing disabled
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
ASSETS = ["BTC-USD", "ETH-USD", "SOL-USD"]
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


def load_asset(symbol):
    print(f"  Loading {symbol} {TIMEFRAME}...")
    try:
        df = yf.download(symbol, interval=TIMEFRAME, period=DATA_PERIOD,
                          auto_adjust=False, progress=False)
    except Exception as e:
        print(f"  [ERROR] Download failed: {e}")
        return None

    if df is None or len(df) == 0:
        print(f"  [ERROR] No data for {symbol}")
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()

    if len(df) < 500:
        print(f"  [SKIP] Only {len(df)} bars, need >= 500")
        return None

    df = add_indicators(df)
    print(f"  {symbol}: {len(df)} bars  "
          f"({df.index[0].date()} -> {df.index[-1].date()})")
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
# K-FOLD VALIDATION FOR ONE ASSET
# =====================================================================
def run_kfold_for_asset(symbol, df):
    n = len(df)
    fold_size = n // N_FOLDS
    boundaries = []
    for k in range(N_FOLDS):
        s = k * fold_size
        e = (k + 1) * fold_size if k < N_FOLDS - 1 else n
        boundaries.append((s, e))

    results = []
    positive_count = 0

    for k, (s, e) in enumerate(boundaries):
        df_fold = df.iloc[s:e].copy()
        curve, trades, bars, pre = run_engine(df_fold, COMMISSION, SLIPPAGE)

        if not trades:
            net = 0.0
            gross = 0.0
            wr = 0.0
            pf = float("nan")
            ic = float("nan")
        else:
            t = pd.DataFrame(trades)
            gross = t["ideal_gross"].sum()
            net = t["net_pnl"].sum()
            wins = t[t["net_pnl"] > 0]["net_pnl"]
            losses = t[t["net_pnl"] <= 0]["net_pnl"]
            wr = 100.0 * len(wins) / len(t)
            pf = (wins.sum() / abs(losses.sum())
                  if losses.sum() != 0 else float("nan"))

            if len(pre) > 50:
                p = np.array([r["pred"] for r in pre], dtype=float)
                f = np.array([r["fwd_ret"] for r in pre], dtype=float)
                if np.std(p) > 1e-9:
                    ic = stats.spearmanr(p, f)[0]
                else:
                    ic = float("nan")
            else:
                ic = float("nan")

        if net > 0:
            positive_count += 1

        results.append({
            "fold": k + 1,
            "bars": e - s,
            "trades": len(trades),
            "gross": gross,
            "net": net,
            "wr": wr,
            "pf": pf,
            "ic": ic,
            "period": f"{df.index[s].date()} -> {df.index[e-1].date()}",
        })

    total_net = sum(r["net"] for r in results)
    total_gross = sum(r["gross"] for r in results)
    total_trades = sum(r["trades"] for r in results)
    mean_net = total_net / N_FOLDS
    std_net = np.std([r["net"] for r in results], ddof=1) if N_FOLDS > 1 else 0.0
    se_net = std_net / np.sqrt(N_FOLDS) if N_FOLDS > 1 else 0.0
    t_stat = mean_net / se_net if se_net > 0 else float("nan")

    return {
        "symbol": symbol,
        "results": results,
        "positive_folds": positive_count,
        "total_net": total_net,
        "total_gross": total_gross,
        "total_trades": total_trades,
        "mean_net": mean_net,
        "std_net": std_net,
        "t_stat": t_stat,
    }


# =====================================================================
# REPORTING
# =====================================================================
def print_asset_folds(asset_result):
    sym = asset_result["symbol"]
    print(f"\n{'=' * 105}")
    print(f"ASSET: {sym}")
    print(f"{'=' * 105}")
    print(f"{'Fold':<6} {'Period':<28} {'Trd':>5} {'Gross':>12} "
          f"{'Net':>12} {'WR%':>8} {'PF':>8} {'IC':>10}")
    print("-" * 105)

    for r in asset_result["results"]:
        pf_str = f"{r['pf']:.3f}" if np.isfinite(r['pf']) else "n/a"
        ic_str = f"{r['ic']:+.5f}" if np.isfinite(r['ic']) else "n/a"
        print(f"{r['fold']:<6} {r['period']:<28} {r['trades']:>5} "
              f"${r['gross']:>11,.0f} ${r['net']:>11,.0f} "
              f"{r['wr']:>7.2f} {pf_str:>8} {ic_str:>10}")

    print("-" * 105)
    print(f"Positive folds:  {asset_result['positive_folds']} / {N_FOLDS}  "
          f"({asset_result['positive_folds'] / N_FOLDS * 100:.1f}%)")
    print(f"Total trades:    {asset_result['total_trades']}")
    print(f"Total gross:     ${asset_result['total_gross']:>12,.2f}")
    print(f"Total net:       ${asset_result['total_net']:>12,.2f}")
    print(f"Mean net/fold:   ${asset_result['mean_net']:>12,.2f}")
    print(f"Std net/fold:    ${asset_result['std_net']:>12,.2f}")
    t_str = f"{asset_result['t_stat']:+.3f}" if np.isfinite(asset_result['t_stat']) else "n/a"
    print(f"t-stat (mean):   {t_str}")


def print_final_summary(all_results):
    print(f"\n{'=' * 105}")
    print(f"CROSS-ASSET SUMMARY")
    print(f"{'=' * 105}")
    print(f"{'Symbol':<12} {'Folds+':>8} {'Trades':>8} "
          f"{'Total Gross':>15} {'Total Net':>15} "
          f"{'Mean/Fold':>12} {'t-stat':>9} {'Verdict':>12}")
    print("-" * 105)

    verdicts = []
    for ar in all_results:
        pos_ratio = ar["positive_folds"] / N_FOLDS
        if pos_ratio >= 0.70 and ar["total_net"] > 0:
            verdict = "STRONG"
        elif pos_ratio >= 0.50 and ar["total_net"] > 0:
            verdict = "MODERATE"
        elif pos_ratio >= 0.33:
            verdict = "WEAK"
        else:
            verdict = "FAIL"
        verdicts.append(verdict)

        t_str = f"{ar['t_stat']:+.3f}" if np.isfinite(ar['t_stat']) else "n/a"
        print(f"{ar['symbol']:<12} "
              f"{ar['positive_folds']}/{N_FOLDS:<6} "
              f"{ar['total_trades']:>8} "
              f"${ar['total_gross']:>14,.0f} "
              f"${ar['total_net']:>14,.0f} "
              f"${ar['mean_net']:>11,.0f} "
              f"{t_str:>9} "
              f"{verdict:>12}")

    # Final verdict
    print(f"\n{'=' * 105}")
    print("FINAL VERDICT")
    print(f"{'=' * 105}")

    strong_count = verdicts.count("STRONG")
    moderate_count = verdicts.count("MODERATE")
    weak_count = verdicts.count("WEAK")
    fail_count = verdicts.count("FAIL")

    print(f"  STRONG:   {strong_count} / {len(verdicts)}")
    print(f"  MODERATE: {moderate_count} / {len(verdicts)}")
    print(f"  WEAK:     {weak_count} / {len(verdicts)}")
    print(f"  FAIL:     {fail_count} / {len(verdicts)}")

    # Aggregate verdict
    positive_assets = strong_count + moderate_count
    if positive_assets >= 2:
        print(f"\n[PASS] Edge appears on {positive_assets}/3 assets.")
        print("       Feature space has cross-asset validity.")
        print("       Next step: parameter stability + live testing.")
    elif positive_assets == 1:
        print(f"\n[WARN] Edge appears on only 1/3 assets.")
        print("       Likely asset-specific or luck.")
        print("       Next step: more assets or longer history.")
    else:
        print(f"\n[FAIL] No asset shows consistent edge.")
        print("       Feature space does not generalize.")
        print("       Recommendation: abandon or redesign features.")


# =====================================================================
# MAIN
# =====================================================================
def main():
    print("=" * 105)
    print("CROSS-ASSET VALIDATION (6-Fold per asset)")
    print("=" * 105)
    print(f"Assets:      {', '.join(ASSETS)}")
    print(f"Timeframe:   {TIMEFRAME}")
    print(f"Folds:       {N_FOLDS}")
    print(f"Config:      MTF={USE_MTF}, kNN on, trailing disabled")
    print(f"Commission:  {COMMISSION * 100:.3f}% per side")
    print(f"Slippage:    {SLIPPAGE * 100:.3f}%")
    print()

    all_results = []
    for sym in ASSETS:
        df = load_asset(sym)
        if df is None:
            print(f"  [SKIP] {sym} - no valid data")
            continue
        asset_result = run_kfold_for_asset(sym, df)
        all_results.append(asset_result)
        print_asset_folds(asset_result)

    if not all_results:
        print("\n[ERROR] No assets could be tested.")
        return

    print_final_summary(all_results)


if __name__ == "__main__":
    main()