import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_dashboard(df):
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.5, 0.15, 0.2, 0.15],
        subplot_titles=[
            "Price + Trend",
            "RSI",
            "MACD",
            "ADX"
        ]
    )

    # -------------------------
    # PRICE (Candlestick)
    # -------------------------
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="Price"
        ),
        row=1, col=1
    )

    # EMA
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['EMA_20'],
            name="EMA 20",
            line=dict(width=1)
        ),
        row=1, col=1
    )

    # VWAP
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['VWAP'],
            name="VWAP",
            line=dict(dash='dash')
        ),
        row=1, col=1
    )

    # Bollinger Bands
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['BB_Upper'],
            name="BB Upper",
            line=dict(width=1),
            opacity=0.4
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['BB_Lower'],
            name="BB Lower",
            fill='tonexty',
            opacity=0.2
        ),
        row=1, col=1
    )

    # -------------------------
    # RSI
    # -------------------------
    fig.add_trace(
        go.Scatter(x=df.index, y=df['RSI'], name="RSI"),
        row=2, col=1
    )

    fig.add_hline(y=70, row=2, col=1)
    fig.add_hline(y=30, row=2, col=1)

    # -------------------------
    # MACD
    # -------------------------
    fig.add_trace(
        go.Scatter(x=df.index, y=df['MACD'], name="MACD"),
        row=3, col=1
    )

    fig.add_trace(
        go.Scatter(x=df.index, y=df['MACD_Signal'], name="Signal"),
        row=3, col=1
    )

    fig.add_trace(
        go.Bar(x=df.index, y=df['MACD_Hist'], name="Histogram"),
        row=3, col=1
    )

    # -------------------------
    # ADX
    # -------------------------
    fig.add_trace(
        go.Scatter(x=df.index, y=df['ADX'], name="ADX"),
        row=4, col=1
    )

    fig.add_hline(y=20, row=4, col=1)

    # -------------------------
    # Layout
    # -------------------------
    fig.update_layout(
        height=1000,
        title="Technical Analysis Dashboard",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        legend=dict(orientation="h")
    )

    fig.show()
