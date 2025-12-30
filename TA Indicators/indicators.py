"""
========================================================
TECHNICAL ANALYSIS INDICATORS USING `ta` LIBRARY
========================================================

This file contains the MOST USED and MOST IMPORTANT
technical indicators used in:
- Trading
- Technical analysis
- Machine learning for finance
- Dash / Plotly dashboards

DATA REQUIREMENTS:
DataFrame `df` must contain:
['Open', 'High', 'Low', 'Close', 'Volume']

--------------------------------------------------------
"""

import ta
import numpy as np

# ======================================================
# TREND INDICATORS
# ======================================================

# ------------------------------------------------------
# SMA – Simple Moving Average
# ------------------------------------------------------
# Measures: Average price over a fixed window
# Use when: Identifying overall trend direction
# How to use:
#   - Price above SMA → bullish
#   - Price below SMA → bearish
#   - Slower than EMA
df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)


# ------------------------------------------------------
# EMA – Exponential Moving Average
# ------------------------------------------------------
# Measures: Trend with more weight on recent prices
# Use when: Faster trend detection (intraday, swing)
# How to use:
#   - EMA reacts quicker than SMA
#   - EMA crossovers = trend change
df['EMA_20'] = ta.trend.ema_indicator(df['Close'], window=20)


# ------------------------------------------------------
# ADX – Average Directional Index
# ------------------------------------------------------
# Measures: Trend strength (NOT direction)
# Values:
#   < 20 → weak / no trend
#   > 20 → strong trend
# How to use:
#   - Trade trend strategies only when ADX > 20
df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=14)

# Directional Indicators
# +DI > -DI → bullish
# -DI > +DI → bearish
df['DI_Pos'] = ta.trend.adx_pos(df['High'], df['Low'], df['Close'], window=14)
df['DI_Neg'] = ta.trend.adx_neg(df['High'], df['Low'], df['Close'], window=14)


# ------------------------------------------------------
# MACD – Moving Average Convergence Divergence
# ------------------------------------------------------
# Measures: Trend + momentum
# How to use:
#   - MACD crossing above signal → bullish
#   - Histogram expansion → momentum strength
df['MACD'] = ta.trend.macd(df['Close'])
df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])
df['MACD_Hist'] = ta.trend.macd_diff(df['Close'])


# ------------------------------------------------------
# ICHIMOKU CLOUD
# ------------------------------------------------------
# Measures: Trend direction, strength, momentum, S/R
# How to use:
#   - Price above cloud → bullish
#   - Thick cloud → strong trend
#   - Tenkan/Kijun cross → entry signal
ichi = ta.trend.IchimokuIndicator(
    df['High'], df['Low'], window1=9, window2=26, window3=52
)

df['Tenkan'] = ichi.ichimoku_conversion_line()
df['Kijun'] = ichi.ichimoku_base_line()
df['Senkou_A'] = ichi.ichimoku_a()
df['Senkou_B'] = ichi.ichimoku_b()


# ======================================================
# MOMENTUM INDICATORS
# ======================================================

# ------------------------------------------------------
# RSI – Relative Strength Index
# ------------------------------------------------------
# Measures: Momentum & overbought/oversold conditions
# Levels:
#   > 70 → overbought
#   < 30 → oversold
# Best used in: Range-bound markets
df['RSI'] = ta.momentum.rsi(df['Close'], window=14)


# ------------------------------------------------------
# STOCHASTIC OSCILLATOR
# ------------------------------------------------------
# Measures: Close price relative to recent range
# Best used: Sideways / ranging markets
df['Stoch_K'] = ta.momentum.stoch(
    df['High'], df['Low'], df['Close'], window=14, smooth_window=3
)
df['Stoch_D'] = ta.momentum.stoch_signal(
    df['High'], df['Low'], df['Close'], window=14, smooth_window=3
)


# ------------------------------------------------------
# CCI – Commodity Channel Index
# ------------------------------------------------------
# Measures: Deviation from mean price
# Use when: Detecting momentum extremes
df['CCI'] = ta.trend.cci(df['High'], df['Low'], df['Close'], window=20)


# ------------------------------------------------------
# ROC – Rate of Change
# ------------------------------------------------------
# Measures: Speed of price movement
# Use when: Momentum-based strategies
df['ROC'] = ta.momentum.roc(df['Close'], window=12)


# ======================================================
# VOLATILITY INDICATORS
# ======================================================

# ------------------------------------------------------
# BOLLINGER BANDS
# ------------------------------------------------------
# Measures: Volatility & mean reversion
# How to use:
#   - Price near upper band → potential pullback
#   - Band squeeze → upcoming breakout
bb = ta.volatility.BollingerBands(df['Close'], window=20)

df['BB_Upper'] = bb.bollinger_hband()
df['BB_Middle'] = bb.bollinger_mavg()
df['BB_Lower'] = bb.bollinger_lband()


# ------------------------------------------------------
# ATR – Average True Range
# ------------------------------------------------------
# Measures: Market volatility (not direction)
# Use when:
#   - Setting stop-loss
#   - Position sizing
df['ATR'] = ta.volatility.average_true_range(
    df['High'], df['Low'], df['Close'], window=14
)


# ------------------------------------------------------
# KELTNER CHANNELS
# ------------------------------------------------------
# Measures: Volatility-based trend channels
# Use when: Trend-following systems
kc = ta.volatility.KeltnerChannel(df['High'], df['Low'], df['Close'])

df['KC_Upper'] = kc.keltner_channel_hband()
df['KC_Lower'] = kc.keltner_channel_lband()


# ======================================================
# VOLUME INDICATORS
# ======================================================

# ------------------------------------------------------
# OBV – On Balance Volume
# ------------------------------------------------------
# Measures: Volume flow
# How to use:
#   - Rising OBV before price → accumulation
#   - Divergence → potential reversal
df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])


# ------------------------------------------------------
# MFI – Money Flow Index
# ------------------------------------------------------
# Measures: RSI + volume
# Levels:
#   > 80 → overbought
#   < 20 → oversold
df['MFI'] = ta.volume.money_flow_index(
    df['High'], df['Low'], df['Close'], df['Volume'], window=14
)


# ------------------------------------------------------
# VWAP – Volume Weighted Average Price
# ------------------------------------------------------
# Measures: Fair price weighted by volume
# Best used: Intraday trading
# How to use:
#   - Price above VWAP → bullish bias
#   - Acts as dynamic support/resistance
df['VWAP'] = ta.volume.volume_weighted_average_price(
    df['High'], df['Low'], df['Close'], df['Volume']
)


# ======================================================
# FINAL CLEANING (CRITICAL)
# ======================================================

# Remove rows created by rolling windows
df.dropna(inplace=True)

# Replace infinity values (safety for ML)
df.replace([np.inf, -np.inf], 0, inplace=True)


"""
========================================================
END OF FILE
========================================================

This file can be used for:
- Plotly dashboards
- Strategy development
- Machine learning feature generation
- Backtesting frameworks
"""
