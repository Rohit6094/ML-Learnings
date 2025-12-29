import pandas as pd
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ta.trend import EMAIndicator, ADXIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.volume import VolumeWeightedAveragePrice


class TradingEngine:

    def __init__(self, df, initial_capital=10000):
        """
        df must contain: Open, High, Low, Close, Volume
        """
        self.df = df.copy()
        self.initial_capital = initial_capital

    def compute_indicators(self):
        df = self.df

        # --------------------
        # Trend Indicators
        # --------------------
        df['EMA_20'] = EMAIndicator(df['Close'], window=20).ema_indicator()

        adx = ADXIndicator(df['High'], df['Low'], df['Close'], window=14)
        df['ADX'] = adx.adx()

        macd = MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Hist'] = macd.macd_diff()

        # --------------------
        # Momentum
        # --------------------
        df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()

        # --------------------
        # Volatility
        # --------------------
        bb = BollingerBands(df['Close'], window=20, window_dev=2)
        df['BB_Upper'] = bb.bollinger_hband()
        df['BB_Lower'] = bb.bollinger_lband()

        # --------------------
        # Volume
        # --------------------
        vwap = VolumeWeightedAveragePrice(
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            volume=df['Volume']
        )
        df['VWAP'] = vwap.volume_weighted_average_price()

        self.df = df

    def generate_signals(self):
        df = self.df

        df['Signal'] = 0

        # BUY
        df.loc[
            (df['Close'] > df['EMA_20']) &
            (df['ADX'] > 20) &
            (df['RSI'] < 70),
            'Signal'
        ] = 1

        # SELL
        df.loc[
            (df['Close'] < df['EMA_20']) |
            (df['RSI'] > 70),
            'Signal'
        ] = -1

        self.df = df

    def backtest(self):
        df = self.df

        df['Returns'] = df['Close'].pct_change()
        df['Strategy_Returns'] = df['Signal'].shift(1) * df['Returns']

        df['Equity'] = (1 + df['Strategy_Returns']).cumprod() * self.initial_capital
        df['BuyHold'] = (1 + df['Returns']).cumprod() * self.initial_capital

        self.df = df

    def performance_metrics(self):
        df = self.df

        total_return = (df['Equity'].iloc[-1] / self.initial_capital) - 1
        max_dd = (df['Equity'] / df['Equity'].cummax() - 1).min()
        sharpe = (
            df['Strategy_Returns'].mean()
            / df['Strategy_Returns'].std()
            * np.sqrt(252)
        )

        return {
            "Total Return": round(total_return * 100, 2),
            "Max Drawdown": round(max_dd * 100, 2),
            "Sharpe Ratio": round(sharpe, 2)
        }

    def plot_dashboard(self):
        df = self.df

        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            row_heights=[0.5, 0.15, 0.2, 0.15],
            subplot_titles=[
                "Price + EMA + VWAP + Bollinger Bands",
                "RSI",
                "MACD",
                "ADX"
            ]
        )

        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close']
            ),
            row=1, col=1
        )

        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], name='EMA 20'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], name='VWAP', line=dict(dash='dash')), row=1, col=1)

        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower', fill='tonexty'), row=1, col=1)

        # RSI
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI'), row=2, col=1)
        fig.add_hline(y=70, row=2, col=1)
        fig.add_hline(y=30, row=2, col=1)

        # MACD
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD'), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal'), row=3, col=1)
        fig.add_bar(x=df.index, y=df['MACD_Hist'], row=3, col=1)

        # ADX
        fig.add_trace(go.Scatter(x=df.index, y=df['ADX'], name='ADX'), row=4, col=1)
        fig.add_hline(y=20, row=4, col=1)

        fig.update_layout(
            height=1000,
            title="Class-Based Trading Engine Dashboard",
            template="plotly_dark",
            xaxis_rangeslider_visible=False
        )

        fig.show()

engine = TradingEngine(df)

engine.compute_indicators()
engine.generate_signals()
engine.backtest()

print(engine.performance_metrics())

engine.plot_dashboard()
