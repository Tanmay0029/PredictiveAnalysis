import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

# keras import guarded to avoid import error if user doesn't need forecast immediately
try:
    from keras.models import load_model
except Exception:
    load_model = None

# =========================
# App Config
# =========================
st.set_page_config(page_title="LSTM Stock Predictor", layout="wide")
st.title("📈 LSTM Stock Price Forecast with Technical Analysis")

MODEL_PATH = "models/Stock_Prediction_Future_Model2.keras"

# Stock lists organized by market
NIFTY_50 = [
    "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS","KOTAKBANK.NS","SBIN.NS",
    "BHARTIARTL.NS","ITC.NS","LT.NS","ASIANPAINT.NS","AXISBANK.NS","BAJFINANCE.NS","ULTRACEMCO.NS",
    "HINDUNILVR.NS","WIPRO.NS","MARUTI.NS","TITAN.NS","SUNPHARMA.NS","JSWSTEEL.NS",
    "TATASTEEL.NS","HCLTECH.NS","POWERGRID.NS","ADANIPORTS.NS","ONGC.NS","COALINDIA.NS",
    "SHREECEM.NS","NESTLEIND.NS","BRITANNIA.NS","EICHERMOT.NS","UPL.NS","BPCL.NS","HEROMOTOCO.NS",
    "GRASIM.NS","DIVISLAB.NS","DRREDDY.NS","M&M.NS","BAJAJFINSV.NS","HDFCLIFE.NS",
    "NTPC.NS","TECHM.NS","TATACONSUM.NS","TATAMOTORS.NS","CIPLA.NS","HINDALCO.NS",
    "ICICIPRULI.NS","APOLLOHOSP.NS","ADANIENT.NS","TATAPOWER.NS"
]

US_STOCKS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "V", "JNJ",
    "WMT", "JPM", "MA", "PG", "XOM", "UNH", "HD", "CVX", "MRK", "KO",
    "ABBV", "PEP", "AVGO", "COST", "LLY", "ADBE", "TMO", "MCD", "CSCO", "ACN"
]

STOCK_GROUPS = {
    "NIFTY-50 (India)": NIFTY_50,
    "US Stocks (Popular)": US_STOCKS
}

# =========================
# Helpers (Indicators)
# =========================
def sma(series, window): 
    return series.rolling(window).mean()

def ema(series, span): 
    return series.ewm(span=span, adjust=False).mean()

def rsi(close, period: int = 14):
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out

def macd(close, fast=12, slow=26, signal=9):
    # Ensure clean series
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.Series(close).squeeze()
    
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    
    return macd_line, signal_line, hist

def bollinger_bands(close, window=20, num_std=2):
    mid = sma(close, window)
    std = close.rolling(window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower

def generate_signals(df):
    # Ensure all indicator columns are clean 1D Series
    for col in ["MACD", "MACD_SIGNAL", "MACD_HIST", "RSI", "Close"]:
        if col in df.columns:
            if isinstance(df[col], pd.DataFrame):
                df[col] = df[col].iloc[:, 0]
            df[col] = pd.Series(df[col]).squeeze()
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Compute MACD cross signals safely
    macd_cross_up = (
        (df["MACD"].shift(1) < df["MACD_SIGNAL"].shift(1)) &
        (df["MACD"] > df["MACD_SIGNAL"])
    )
    
    macd_cross_down = (
        (df["MACD"].shift(1) > df["MACD_SIGNAL"].shift(1)) &
        (df["MACD"] < df["MACD_SIGNAL"])
    )
    
    # Create clean 1D output arrays
    buy_signal = np.where(macd_cross_up & (df["RSI"] < 60), df["Close"], np.nan)
    sell_signal = np.where(macd_cross_down & (df["RSI"] > 40), df["Close"], np.nan)
    
    df["BUY_SIGNAL"] = pd.Series(buy_signal, index=df.index)
    df["SELL_SIGNAL"] = pd.Series(sell_signal, index=df.index)
    
    return df

# =========================
# Cache layers
# =========================
@st.cache_data(show_spinner=False)
def fetch_data(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    
    # Handle multi-level columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    
    df = df.reset_index()
    
    # Ensure all price columns are 1D Series
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            if isinstance(df[col], pd.DataFrame):
                df[col] = df[col].iloc[:, 0]
            df[col] = pd.Series(df[col]).squeeze()
    
    return df

@st.cache_resource(show_spinner=False)
def load_lstm_model_cached(path: str):
    if not os.path.exists(path):
        return None
    if load_model is None:
        return None
    return load_model(path)

# =========================
# Sidebar Controls
# =========================
with st.sidebar:
    st.header("⚙️ Controls")
    
    # Market selection
    market = st.selectbox("Select Market", list(STOCK_GROUPS.keys()))
    stock_list = STOCK_GROUPS[market]
    
    stock = st.selectbox("Select Stock", stock_list, index=0)
    start_date = st.date_input("Start Date", datetime(2020, 1, 1))
    end_date = st.date_input("End Date", datetime.now())
    future_days = st.slider("Future Days to Predict", 1, 365, 30)
    
    chart_type = st.radio("Chart Type", ["Candlestick", "Line"], horizontal=True)
    show_volume = st.toggle("Show Volume", value=True)
    indicator_opts = st.multiselect(
        "Indicators",
        ["SMA_50", "EMA_20", "EMA_12_26", "RSI", "MACD", "Bollinger Bands"],
        default=["SMA_50", "EMA_20", "RSI", "MACD"]
    )
    
    st.divider()
    refresh = st.button("🔄 Refresh Data")

# =========================
# Data load
# =========================
if refresh:
    st.toast("Fetching latest data...", icon="⏳")

df = fetch_data(stock, start_date, end_date)
if df.empty:
    st.error("No data for the chosen range. Try different dates.")
    st.stop()

# Ensure Close is a clean numeric 1D Series
if isinstance(df["Close"], pd.DataFrame):
    df["Close"] = df["Close"].iloc[:, 0]
df["Close"] = pd.Series(df["Close"]).squeeze()
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

st.success(f"✅ Data Loaded: {len(df)} rows")

# =========================
# Compute indicators
# =========================
df["SMA_50"] = sma(df["Close"], 50)
df["EMA_20"] = ema(df["Close"], 20)

df["RSI"] = rsi(df["Close"], period=14)
macd_line, signal_line, hist = macd(df["Close"], fast=12, slow=26, signal=9)
df["MACD"], df["MACD_SIGNAL"], df["MACD_HIST"] = macd_line, signal_line, hist

bb_u, bb_m, bb_l = bollinger_bands(df["Close"], window=20, num_std=2)
df["BB_UPPER"], df["BB_MID"], df["BB_LOWER"] = bb_u, bb_m, bb_l

df = generate_signals(df)

# =========================
# Forecast (if model present)
# =========================
model = load_lstm_model_cached(MODEL_PATH)
pred_df = None
if model is None:
    warn_msg = "⚠️ Trained model not found or unavailable. Put it at `models/Stock_Prediction_Future_Model2.keras` to enable forecasting."
    st.info(warn_msg)
else:
    # scale only on Close for forecasting
    close_values = df["Close"].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(close_values)
    
    if len(scaled) >= 100:
        last_100 = scaled[-100:].flatten().tolist()
        seq = last_100[:]
        preds = []
        for _ in range(future_days):
            x = np.array(seq[-100:]).reshape(1, 100, 1)
            y = model.predict(x, verbose=0)[0][0]
            preds.append(y)
            seq.append(y)
        
        preds = np.array(preds).reshape(-1, 1)
        preds = preds / scaler.scale_
        
        last_date = pd.to_datetime(df["Date"].iloc[-1])
        future_dates = pd.date_range(last_date + timedelta(days=1), periods=future_days)
        
        pred_df = pd.DataFrame({"Date": future_dates, "Predicted": preds.flatten()})
    else:
        st.warning("Not enough data to seed the LSTM (needs at least 100 closing prices).")

# =========================
# Tabs
# =========================
tab_overview, tab_tech, tab_forecast, tab_data = st.tabs(
    ["📊 Overview", "🧮 Technicals", "🔮 Forecast", "📜 Data"]
)

# =========================
# Overview (Main Chart)
# =========================
with tab_overview:
    st.subheader(f"{stock} — Price Chart")
    
    fig = go.Figure()
    
    # Price
    if chart_type == "Candlestick":
        fig.add_candlestick(
            x=df["Date"],
            open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"],
            name="Candlestick"
        )
    else:
        fig.add_trace(go.Scatter(
            x=df["Date"], y=df["Close"],
            mode="lines", name="Close", line=dict(width=2)
        ))
    
    # Indicators overlay
    if "SMA_50" in indicator_opts:
        fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA_50"], mode="lines", name="SMA 50"))
    if "EMA_20" in indicator_opts:
        fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA_20"], mode="lines", name="EMA 20"))
    if "Bollinger Bands" in indicator_opts:
        fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_UPPER"], mode="lines", name="BB Upper", line=dict(width=1)))
        fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_MID"], mode="lines", name="BB Mid", line=dict(width=1, dash="dot")))
        fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_LOWER"], mode="lines", name="BB Lower", line=dict(width=1)))
    
    # Buy/Sell markers
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["BUY_SIGNAL"],
        mode="markers", name="Buy",
        marker=dict(symbol="triangle-up", size=10, color="green")
    ))
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["SELL_SIGNAL"],
        mode="markers", name="Sell",
        marker=dict(symbol="triangle-down", size=10, color="red")
    ))
    
    # Predicted dotted line
    if pred_df is not None:
        fig.add_trace(go.Scatter(
            x=pred_df["Date"], y=pred_df["Predicted"],
            mode="lines", name="Predicted",
            line=dict(dash="dot", width=3, color="orange")
        ))
    
    # Volume (separate axis)
    if show_volume:
        fig.add_trace(go.Bar(
            x=df["Date"], y=df["Volume"],
            name="Volume", opacity=0.25, yaxis="y2"
        ))
        fig.update_layout(
            yaxis2=dict(overlaying='y', side='right', showgrid=False, title="Volume", rangemode='tozero')
        )
    
    fig.update_layout(
        template="plotly_dark",
        height=650,
        legend_title="Legend",
        xaxis_title="Date",
        yaxis_title="Price"
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================
# Technicals (RSI, MACD panels)
# =========================
with tab_tech:
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("### RSI (14)")
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df["Date"], y=df["RSI"], mode="lines", name="RSI"))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
        fig_rsi.update_layout(template="plotly_dark", height=300, yaxis_title="RSI", xaxis_title="Date")
        st.plotly_chart(fig_rsi, use_container_width=True)
    
    with col2:
        st.markdown("### MACD (12, 26, 9)")
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df["Date"], y=df["MACD"], mode="lines", name="MACD"))
        fig_macd.add_trace(go.Scatter(x=df["Date"], y=df["MACD_SIGNAL"], mode="lines", name="Signal"))
        fig_macd.add_trace(go.Bar(x=df["Date"], y=df["MACD_HIST"], name="Hist", opacity=0.5))
        fig_macd.update_layout(template="plotly_dark", height=300, xaxis_title="Date")
        st.plotly_chart(fig_macd, use_container_width=True)

# =========================
# Forecast Tab
# =========================
with tab_forecast:
    st.markdown("### Future Predictions")
    if pred_df is None:
        st.info("Forecasting disabled (model missing). Add the .keras model to enable this tab.")
    else:
        st.dataframe(pred_df, use_container_width=True, height=400)
        
        # Download predictions
        out_csv = pred_df.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Download Predictions (CSV)",
            data=out_csv,
            file_name=f"{stock}_predictions_{future_days}d.csv",
            mime="text/csv"
        )

# =========================
# Data Tab
# =========================
with tab_data:
    st.markdown("### Historical Data")
    st.dataframe(df, use_container_width=True, height=420)
    
    hist_csv = df.to_csv(index=False).encode()
    st.download_button(
        "⬇️ Download Historical Data (CSV)",
        data=hist_csv,
        file_name=f"{stock}_historical_{start_date}_{end_date}.csv",
        mime="text/csv"
    )

# Footer
st.caption("Built with ❤️ by Tanmay • Plotly/Streamlit • NIFTY-50 & US Stocks • LSTM forecast")