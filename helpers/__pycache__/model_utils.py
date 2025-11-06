import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta

def load_lstm_model(path):
    return load_model(path)


def predict_future_prices(model, data, future_days=30):
    close_prices = data["Close"]

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(close_prices.values.reshape(-1,1))

    # last 100 values for prediction
    last_100 = scaled_data[-100:]
    input_seq = list(last_100.reshape(100))

    future_output = []

    for _ in range(future_days):
        x_input = np.array(input_seq[-100:]).reshape(1, 100, 1)
        pred = model.predict(x_input, verbose=0)[0][0]
        future_output.append(pred)
        input_seq.append(pred)

    future_output = np.array(future_output).reshape(-1, 1)
    future_output = future_output / scaler.scale_

    # Create dates
    last_date = data["Date"].iloc[-1]
    future_dates = pd.date_range(last_date + timedelta(days=1), periods=future_days)

    return pd.DataFrame({
        "Date": future_dates,
        "Predicted Price": future_output.flatten()
    })
