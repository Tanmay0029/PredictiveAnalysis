# 📈 PredictiveAnalysis  
> LSTM-powered stock-price forecasting + advanced technical analytics

**PredictiveAnalysis** is a sleek, interactive dashboard that brings together deep-learning and financial technical analysis to help you **forecast future stock prices** (for NIFTY 50 stocks) and visualise key indicators with elegance.

---

## ✨ Why this project matters  
- Long-term investment and trading decisions benefit from both **historical patterns** and **future predictions**.  
- Many tools show past data only — this tool pushes further: **predicting up to 365 days ahead**.  
- Combines a **trained LSTM model** with **moving averages, RSI, MACD, Bollinger Bands**, and even **buy/sell-signal markers**.  
- All packed into a clean, user-friendly **Streamlit app** with live-data integration via Yahoo Finance.

---

## 🛠 Features at a glance  
| Feature | Description |
|---------|-------------|
| NIFTY 50 Dropdown | Choose any stock from the Nifty 50 list (hard-coded) to analyse. |
| Custom Future-Days Slider | Predict **1–365 days** into the future with ease. |
| Interactive Charts | Candlestick or line charts with dark theme, zoom/hover, and toggle volume. |
| Technical Indicators | SMA 50, EMA 20, RSI, MACD (12/26/9), Bollinger Bands — select per your preference. |
| Buy/Sell Markers | Visual markers indicate MACD-cross + RHS filters for intuitive signals. |
| Forecast Tab | Dedicated tab showing future-predicted values and downloadable CSV. |
| Downloadable Data | Export both historical and predicted data as CSV. |
| Smooth UX | Caching, friendly fallback if the model is missing, and intuitive controls. |

---

## 📂 Project Structure  
```

PredictiveAnalysis/
├── app1.py                ← Main Streamlit application
├── helpers/               ← Utility modules
│   ├── indicators.py
│   └── model_utils.py
├── models/                ← Saved trained LSTM model
│   └── Stock_Prediction_Future_Model2.keras
├── requirements.txt       ← Dependencies list
└── README.md              ← This file

````

---

## 🚀 Getting Started

### 1. Clone the repository  
```bash
git clone https://github.com/Tanmay0029/PredictiveAnalysis.git  
cd PredictiveAnalysis
````

### 2. Create & activate a virtual environment

```bash
python -m venv venv  
# Windows
venv\Scripts\activate  
# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the trained model

Ensure you have the model file placed at:

```
models/Stock_Prediction_Future_Model2.keras
```

If the file is missing, the app will still load — but forecasts will be disabled.

### 5. Run the Streamlit app

```bash
streamlit run app1.py
```

Open the URL displayed in your browser to interact with the dashboard.

---

## 🎯 How it works

1. The app uses yfinance to fetch live stock data based on the selected symbol and date range.
2. Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands) are calculated in real-time.
3. If the LSTM model is available, the last 100 closing prices are scaled and fed into the model to generate future price predictions up to the selected number of days.
4. The chart visualises both the historical data and predicted future data as a dotted line, along with the selected indicators and buy/sell markers.
5. Users can export both historical and predicted data as CSV files for further analysis.

---

## 📌 Use Cases

* New traders who want both technical signals and a forecast to guide decisions.
* Data scientists exploring time-series and LSTM modelling for financial data.
* Investors who prefer a visualised holistic view of past, present & future price trends.
* Educators demonstrating the real-world application of deep learning + finance.

---

## 🔧 Future Enhancements

* Multi-stock comparison (overlay two or more symbols)
* PDF or Excel report generation (with charts & signals summary)
* Option to retrain or fine-tune the model from the dashboard
* Integration of real‐time news sentiment or social media signals
* Back-testing module with historical signal performance metrics

---

## 📜 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). Feel free to fork, enhance, and share!

---

## ❤️ Thanks & Connect

Built with ❤️ by Tanmay.
If you like the project, please ★ star it on GitHub, share your feedback or open pull requests.
Let’s push the frontier of predictive analytics together.

---
