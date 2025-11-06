import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_rsi(data, period=14):
    delta = data["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def plot_ma(data):
    plt.figure(figsize=(10, 5))
    plt.plot(data["Close"], label="Close", color="black")
    plt.plot(data["Close"].rolling(100).mean(), label="MA 100", color="red")
    plt.plot(data["Close"].rolling(200).mean(), label="MA 200", color="blue")
    plt.legend()
    plt.grid(True)
    return plt.gcf()


def plot_rsi(data):
    plt.figure(figsize=(10, 3))
    plt.plot(data["RSI"], label="RSI", color="purple")
    plt.axhline(70, linestyle="--", color="red")
    plt.axhline(30, linestyle="--", color="green")
    plt.legend()
    plt.grid(True)
    return plt.gcf()
