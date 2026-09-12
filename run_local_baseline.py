# run_local_baseline.py

import numpy as np
import pandas as pd
import math

#=====================================================================
# 1. BASELINE CONFIGURATION
#=====================================================================

SYMBOL = "BTC-USD"
TIMEFRAME = "1h"
DATA_PERIOD = "730d"          # yfinance allows about 730 days for 1h crypto data

INITIAL_CAPITAL = 100_000.0
POSITION_PCT = 0.10           # 10% of equity notional per trade
COMMISSION = 0.0005           # 0.05% per side
SLIPPAGE = 0.0002             # 0.02% adverse slippage

# Causal kNN baseline parameters
PATTERN_LEN = 10
MEMORY_SIZE = 80
K_NEIGHBORS = 5
AHEAD = 2
PRED_SMOOTH = 4
MIN_PRED_ATR = 0.20

# Regime filters
USE_MTF = True
HTF_TIMEFRAME = "4h"          # 240 minutes
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

# Risk management
STOP_ATR = 2.0
TP_ATR = 3.5
TRAIL_ATR = 2.0

# Annualization factor for 1h crypto
PERIODS_PER_YEAR = 24 * 365


#=====================================================================
# 2. DATA LOADING
#=====================================================================

def load_data():
    try:
        import yfinance as yf
    except ImportError:
        raise SystemExit(
            "yfinance is not installed. Please run: pip install yfinance"
        )

    print(f"Downloading {SYMBOL} {TIMEFRAME} data...")

    df = yf.download(
        SYMBOL,
        interval=TIMEFRAME,
        period=DATA_PERIOD,
        auto_adjust=False,
        progress=False
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)

    if len(df) < 1000:
        raise SystemExit(
            "Not enough data downloaded. Check internet access / symbol."
        )

    print(f"Downloaded {len(df)} bars.")
    return df


#=====================================================================
# 3. INDICATOR HELPERS
#=====================================================================

def wilder(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:

    # True Range and ATR
    prev_close = df["Close"].shift(1)

    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs()
        ],
        axis=1
    ).max(axis=1)

    df["ATR"] = wilder(tr, ATR_LEN)
    df["ATR_PCT"] = np.where(
        df["Close"] > 0,
        df["ATR"] / df["Close"] * 100.0,
        np.nan
    )

    # EMA fast/slow
    df["EMA_FAST"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["EMA_SLOW"] = df["Close"].ewm(span=30, adjust=False).mean()

    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = wilder(gain, 14)
    avg_loss = wilder(loss, 14)

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    df["RSI"] = 100.0 - (100.0 / (1.0 + rs))
    df["RSI"] = df["RSI"].fillna(50.0)

    # Bollinger Bandwidth percentage
    bb_basis = df["Close"].rolling(BB_LEN).mean()
    bb_std = df["Close"].rolling(BB_LEN).std(ddof=1)

    df["BB_WIDTH_PCT"] = np.where(
        bb_basis > 0,
        (2.0 * BB_MULT * bb_std / bb_basis) * 100.0,
        np.nan
    )

    # Volume confirmation
    df["VOL_SMA"] = df["Volume"].rolling(VOLUME_LEN).mean()
    df["VOL_CONF"] = (
        df["Volume"].notna()
        & df["VOL_SMA"].notna()
        & (df["Volume"] > df["VOL_SMA"] * VOLUME_MULT)
    )

    # ADX / DMI
    up = df["High"].diff()
    down = -df["Low"].diff()

    plus_dm = pd.Series(
        np.where((up > down) & (up > 0), up, 0.0),
        index=df.index
    )

    minus_dm = pd.Series(
        np.where((down > up) & (down > 0), down, 0.0),
        index=df.index
    )

    plus_dm_smooth = wilder(plus_dm, ADX_LEN)
    minus_dm_smooth = wilder(minus_dm, ADX_LEN)

    atr_safe = df["ATR"].replace(0.0, np.nan)

    di_plus = 100.0 * plus_dm_smooth / atr_safe
    di_minus = 100.0 * minus_dm_smooth / atr_safe

    di_sum = (di_plus + di_minus).replace(0.0, np.nan)
    dx = 100.0 * (di_plus - di_minus).abs() / di_sum

    df["ADX"] = wilder(dx.fillna(0.0), ADX_LEN)

    # Higher timeframe confirmed EMA
    htf_close = df["Close"].resample(HTF_TIMEFRAME).last()
    htf_ema = htf_close.ewm(span=HTF_EMA_LEN, adjust=False).mean()

    # Shift by one HTF bar to emulate confirmed HTF data.
    htf_ema_confirmed = htf_ema.shift(1).reindex(df.index, method="ffill")

    df["HTF_EMA"] = htf_ema_confirmed
    df["HTF_BULL"] = df["Close"] > df["HTF_EMA"]
    df["HTF_BEAR"] = df["Close"] < df["HTF_EMA"]

    # Feature engineering: ATR-normalized, clipped
    vol_norm = np.maximum(df["ATR"] / df["Close"], 1e-6)

    f1 = np.log(df["Close"] / df["Close"].shift(1)) / vol_norm
    f2 = (df["Close"] - df["Close"].shift(5)) / df["ATR"]
    f3 = (df["RSI"] - 50.0) / 50.0
    f4 = (df["EMA_FAST"] - df["EMA_SLOW"]) / df["ATR"]

    df["f1"] = np.clip(
        f1.replace([np.inf, -np.inf], 0.0).fillna(0.0),
        -5.0,
        5.0
    )

    df["f2"] = np.clip(
        f2.replace([np.inf, -np.inf], 0.0).fillna(0.0),
        -5.0,
        5.0
    )

    df["f3"] = np.clip(
        f3.replace([np.inf, -np.inf], 0.0).fillna(0.0),
        -5.0,
        5.0
    )

    df["f4"] = np.clip(
        f4.replace([np.inf, -np.inf], 0.0).fillna(0.0),
        -5.0,
        5.0
    )

    return df


#=====================================================================
# 4. LOCAL BACKTEST ENGINE
#=====================================================================

def run_backtest(df: pd.DataFrame):

    open_ = df["Open"].values.astype(float)
    high = df["High"].values.astype(float)
    low = df["Low"].values.astype(float)
    close = df["Close"].values.astype(float)

    atr = df["ATR"].values.astype(float)
    atr_pct = df["ATR_PCT"].values.astype(float)

    f1 = df["f1"].values.astype(float)
    f2 = df["f2"].values.astype(float)
    f3 = df["f3"].values.astype(float)
    f4 = df["f4"].values.astype(float)

    htf_bull = df["HTF_BULL"].fillna(False).astype(bool).values
    htf_bear = df["HTF_BEAR"].fillna(False).astype(bool).values

    bbw = df["BB_WIDTH_PCT"].values.astype(float)
    adx = df["ADX"].values.astype(float)
    vol_conf = df["VOL_CONF"].fillna(False).astype(bool).values

    n = len(df)

    vec_len = PATTERN_LEN * 4

    # Training memory ring buffer
    X = np.zeros((MEMORY_SIZE, vec_len), dtype=float)
    Y = np.zeros(MEMORY_SIZE, dtype=float)

    head = 0
    count = 0

    # Prediction EMA state
    pred_ema = None
    ema_alpha = 2.0 / (PRED_SMOOTH + 1.0) if PRED_SMOOTH > 1 else 1.0

    # Trading state
    equity = INITIAL_CAPITAL
    position = 0
    qty = 0.0

    entry_price = 0.0
    entry_i = None
    entry_comm = 0.0

    initial_stop = 0.0
    trail_stop = 0.0
    target = 0.0

    pending_signal = 0
    pending_atr = np.nan

    equity_curve = np.zeros(n, dtype=float)
    trades = []
    bars_in_market = 0

    required = max(
        350,
        PATTERN_LEN + AHEAD + 50,
        HTF_EMA_LEN * 4 + 50
    )

    #---------------------------------------------------------------
    # Nested exit helper
    #---------------------------------------------------------------
    def exit_position(i: int, exit_price_raw: float, reason: str):
        nonlocal equity, position, qty, entry_price, entry_i, entry_comm
        nonlocal initial_stop, trail_stop, target

        if position == 0 or qty <= 0.0:
            return

        if position == 1:
            fill = exit_price_raw * (1.0 - SLIPPAGE)
            gross = (fill - entry_price) * qty
        else:
            fill = exit_price_raw * (1.0 + SLIPPAGE)
            gross = (entry_price - fill) * qty

        exit_comm = fill * qty * COMMISSION

        # Entry commission was already deducted when entering.
        equity += gross - exit_comm

        net_trade_pnl = gross - exit_comm - entry_comm

        trades.append(
            {
                "entry_time": df.index[entry_i],
                "exit_time": df.index[i],
                "dir": "long" if position == 1 else "short",
                "pnl": net_trade_pnl,
                "bars_held": i - entry_i,
                "reason": reason
            }
        )

        position = 0
        qty = 0.0
        entry_price = 0.0
        entry_i = None
        entry_comm = 0.0
        initial_stop = 0.0
        trail_stop = 0.0
        target = 0.0

    #---------------------------------------------------------------
    # Main event loop
    #---------------------------------------------------------------
    for i in range(n):

        #===========================================================
        # A. Execute pending signal from previous confirmed bar
        #===========================================================
        if pending_signal != 0 and i > 0:

            # Close/reverse existing opposite position at open
            if position != 0 and position != pending_signal:
                exit_position(i, open_[i], "signal_reverse")

            # Open new position if flat
            if position == 0:

                direction = pending_signal
                entry_atr = pending_atr

                if not np.isfinite(entry_atr) or entry_atr <= 0:
                    entry_atr = atr[i]

                if np.isfinite(entry_atr) and entry_atr > 0 and equity > 0:

                    if direction == 1:
                        fill = open_[i] * (1.0 + SLIPPAGE)
                    else:
                        fill = open_[i] * (1.0 - SLIPPAGE)

                    qty = (equity * POSITION_PCT) / fill
                    entry_price = fill
                    entry_i = i
                    entry_comm = fill * qty * COMMISSION

                    equity -= entry_comm

                    if direction == 1:
                        initial_stop = entry_price - STOP_ATR * entry_atr
                        target = entry_price + TP_ATR * entry_atr
                    else:
                        initial_stop = entry_price + STOP_ATR * entry_atr
                        target = entry_price - TP_ATR * entry_atr

                    trail_stop = initial_stop
                    position = direction

            pending_signal = 0
            pending_atr = np.nan

        #===========================================================
        # B. Check stops / take profits intrabar
        #===========================================================
        if position != 0:

            if position == 1:

                effective_stop = max(initial_stop, trail_stop)

                if open_[i] <= effective_stop:
                    exit_position(i, open_[i], "stop")
                elif low[i] <= effective_stop:
                    exit_position(i, effective_stop, "stop")
                elif open_[i] >= target:
                    exit_position(i, open_[i], "tp")
                elif high[i] >= target:
                    exit_position(i, target, "tp")

            else:

                effective_stop = min(initial_stop, trail_stop)

                if open_[i] >= effective_stop:
                    exit_position(i, open_[i], "stop")
                elif high[i] >= effective_stop:
                    exit_position(i, effective_stop, "stop")
                elif open_[i] <= target:
                    exit_position(i, open_[i], "tp")
                elif low[i] <= target:
                    exit_position(i, target, "tp")

        #===========================================================
        # C. Update trailing stop after confirmed close
        #===========================================================
        if position != 0 and np.isfinite(atr[i]) and atr[i] > 0:

            if position == 1:
                trail_candidate = close[i] - TRAIL_ATR * atr[i]
                trail_stop = max(trail_stop, trail_candidate)
            else:
                trail_candidate = close[i] + TRAIL_ATR * atr[i]
                trail_stop = min(trail_stop, trail_candidate)

        if position != 0:
            bars_in_market += 1

        #===========================================================
        # D. Causal kNN model update at confirmed close
        #===========================================================
        raw_prediction = None

        if i >= required:

            #-------------------------------------------------------
            # 1. Add mature training sample:
            #    pattern ending at i - AHEAD
            #    label = log(close[i] / close[i - AHEAD])
            #-------------------------------------------------------
            train_end = i - AHEAD

            if train_end >= PATTERN_LEN - 1:

                start = train_end - PATTERN_LEN + 1
                end = train_end + 1

                vec = np.concatenate(
                    [
                        f1[start:end],
                        f2[start:end],
                        f3[start:end],
                        f4[start:end]
                    ]
                )

                if close[i] > 0 and close[train_end] > 0:

                    label = math.log(close[i] / close[train_end])

                    write_idx = head % MEMORY_SIZE
                    X[write_idx] = vec
                    Y[write_idx] = label

                    head += 1
                    count = min(count + 1, MEMORY_SIZE)

            #-------------------------------------------------------
            # 2. Build current query pattern ending at bar i
            #-------------------------------------------------------
            if i >= PATTERN_LEN - 1:

                start = i - PATTERN_LEN + 1
                end = i + 1

                query = np.concatenate(
                    [
                        f1[start:end],
                        f2[start:end],
                        f3[start:end],
                        f4[start:end]
                    ]
                )

                if count >= K_NEIGHBORS:

                    if count < MEMORY_SIZE:
                        valid_X = X[:count]
                        valid_Y = Y[:count]
                    else:
                        valid_X = X
                        valid_Y = Y

                    distances = np.linalg.norm(valid_X - query, axis=1)

                    if K_NEIGHBORS < len(distances):
                        top = np.argpartition(distances, K_NEIGHBORS - 1)[:K_NEIGHBORS]
                    else:
                        top = np.argsort(distances)

                    d_top = distances[top]
                    y_top = valid_Y[top]

                    weights = 1.0 / (d_top + 1e-6)
                    weight_sum = weights.sum()

                    if weight_sum > 0:
                        raw_prediction = float((weights * y_top).sum() / weight_sum)

            #-------------------------------------------------------
            # 3. Smooth prediction
            #-------------------------------------------------------
            if raw_prediction is not None:
                if pred_ema is None:
                    pred_ema = raw_prediction
                else:
                    pred_ema = ema_alpha * raw_prediction + (1.0 - ema_alpha) * pred_ema

        #===========================================================
        # E. Signal generation for next bar
        #===========================================================
        if (
            i < n - 1
            and pred_ema is not None
            and np.isfinite(atr_pct[i])
            and atr_pct[i] > 0
        ):

            prediction_score = pred_ema / (atr_pct[i] / 100.0)

            volatility_regime = (
                np.isfinite(atr_pct[i])
                and np.isfinite(bbw[i])
                and atr_pct[i] >= MIN_ATR_PCT
                and bbw[i] >= MIN_BB_WIDTH_PCT
            )

            adx_regime = (
                np.isfinite(adx[i])
                and adx[i] >= MIN_ADX
            )

            long_filters = (
                (not USE_MTF or htf_bull[i])
                and (not USE_VOLATILITY or volatility_regime)
                and (not USE_VOLUME or vol_conf[i])
                and (not USE_ADX or adx_regime)
            )

            short_filters = (
                (not USE_MTF or htf_bear[i])
                and (not USE_VOLATILITY or volatility_regime)
                and (not USE_VOLUME or vol_conf[i])
                and (not USE_ADX or adx_regime)
            )

            if (
                prediction_score >= MIN_PRED_ATR
                and long_filters
                and position <= 0
            ):
                pending_signal = 1
                pending_atr = atr[i]

            elif (
                prediction_score <= -MIN_PRED_ATR
                and short_filters
                and position >= 0
            ):
                pending_signal = -1
                pending_atr = atr[i]

        #===========================================================
        # F. Mark-to-market equity curve
        #===========================================================
        unrealized = 0.0

        if position == 1:
            unrealized = (close[i] - entry_price) * qty
        elif position == -1:
            unrealized = (entry_price - close[i]) * qty

        equity_curve[i] = equity + unrealized

    # Close any open position at final close for trade statistics
    if position != 0:
        exit_position(n - 1, close[-1], "end_of_data")
        equity_curve[-1] = equity

    return df, equity_curve, trades, bars_in_market


#=====================================================================
# 5. METRICS
#=====================================================================

def calculate_metrics(
    df: pd.DataFrame,
    equity_curve: np.ndarray,
    trades: list,
    bars_in_market: int
):

    equity = pd.Series(equity_curve, index=df.index)

    net_profit = equity.iloc[-1] - INITIAL_CAPITAL
    net_profit_pct = net_profit / INITIAL_CAPITAL * 100.0

    returns = equity.pct_change().fillna(0.0)

    if returns.std() > 0:
        sharpe = returns.mean() / returns.std() * math.sqrt(PERIODS_PER_YEAR)
    else:
        sharpe = 0.0

    running_max = equity.cummax()
    drawdown_pct = (equity / running_max - 1.0) * 100.0
    max_dd_pct = drawdown_pct.min()

    trades_df = pd.DataFrame(trades)

    if len(trades_df) == 0:
        return {
            "Total Trades": 0,
            "Net Profit": net_profit,
            "Net Profit %": net_profit_pct,
            "Profit Factor": 0.0,
            "Win Rate %": 0.0,
            "Max Drawdown %": max_dd_pct,
            "Sharpe": sharpe,
            "Average Trade": 0.0,
            "Average Bars Held": 0.0,
            "Max Consecutive Losses": 0,
            "Percent Time In Market": 0.0
        }

    pnls = trades_df["pnl"].values

    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    gross_win = wins.sum() if len(wins) > 0 else 0.0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 0.0

    if gross_loss > 0:
        profit_factor = gross_win / gross_loss
    else:
        profit_factor = float("inf") if gross_win > 0 else 0.0

    win_rate = 100.0 * len(wins) / len(pnls)

    max_consec_losses = 0
    current = 0

    for pnl in pnls:
        if pnl < 0:
            current += 1
            max_consec_losses = max(max_consec_losses, current)
        else:
            current = 0

    percent_time_in_market = 100.0 * bars_in_market / len(df)

    return {
        "Total Trades": len(trades_df),
        "Net Profit": net_profit,
        "Net Profit %": net_profit_pct,
        "Profit Factor": profit_factor,
        "Win Rate %": win_rate,
        "Max Drawdown %": max_dd_pct,
        "Sharpe": sharpe,
        "Average Trade": pnls.mean(),
        "Average Bars Held": trades_df["bars_held"].mean(),
        "Max Consecutive Losses": max_consec_losses,
        "Percent Time In Market": percent_time_in_market
    }


#=====================================================================
# 6. MAIN
#=====================================================================

def main():

    df = load_data()
    df = add_indicators(df)

    print("Running local causal kNN backtest...")

    df, equity_curve, trades, bars_in_market = run_backtest(df)

    metrics = calculate_metrics(df, equity_curve, trades, bars_in_market)

    print("=" * 60)
    print("LOCAL PYTHON BASELINE BACKTEST")
    print("=" * 60)
    print(f"Symbol:                  {SYMBOL}")
    print(f"Timeframe:               {TIMEFRAME}")
    print(f"Data period:             {df.index[0]} to {df.index[-1]}")
    print(f"Bars:                    {len(df)}")
    print("-" * 60)
    print(f"Initial Capital:         {INITIAL_CAPITAL:,.2f}")
    print(f"Commission:              {COMMISSION * 100:.3f}% per side")
    print(f"Slippage:                {SLIPPAGE * 100:.3f}%")
    print(f"Position Size:           {POSITION_PCT * 100:.1f}% equity notional")
    print("-" * 60)
    print(f"Total Trades:            {metrics['Total Trades']}")
    print(f"Net Profit:              {metrics['Net Profit']:,.2f}")
    print(f"Net Profit %:            {metrics['Net Profit %']:.2f}%")
    print(f"Profit Factor:           {metrics['Profit Factor']:.3f}")
    print(f"Win Rate:                {metrics['Win Rate %']:.2f}%")
    print(f"Max Drawdown:            {metrics['Max Drawdown %']:.2f}%")
    print(f"Sharpe Ratio:            {metrics['Sharpe']:.2f}")
    print(f"Average Trade PnL:       {metrics['Average Trade']:,.2f}")
    print(f"Average Bars Held:       {metrics['Average Bars Held']:.2f}")
    print(f"Max Consecutive Losses:  {metrics['Max Consecutive Losses']}")
    print(f"Percent Time In Market:  {metrics['Percent Time In Market']:.2f}%")
    print("=" * 60)
    # Final equity برای cross-check
    final_equity = equity_curve[-1]
    print(f"\nFINAL EQUITY: ${final_equity:,.2f}")
    print(f"NET PnL:      ${final_equity - INITIAL_CAPITAL:,.2f}")
    print("\nPaste the output above as the real baseline metrics.")


if __name__ == "__main__":
    main()