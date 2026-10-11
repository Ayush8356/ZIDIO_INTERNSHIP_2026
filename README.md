"""
Time Series Stock Forecasting (AAPL): ARIMA vs Prophet vs LSTM

Pipeline
1. Download AAPL daily closing prices (yfinance)
2. Chronological split: everything except the last TEST_DAYS business days is
   training data, the last TEST_DAYS business days are the hold-out test set
3. Fit each model on TRAIN ONLY and forecast the exact test dates
4. Compare against a naive baseline (last known price) using RMSE, MAE, MAPE
5. Save plots and a metrics table

Run: python main.py
"""

import logging
import random
import warnings

import matplotlib

matplotlib.use("Agg")  # save plots to files, no pop-up windows
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Sequential
import tensorflow as tf

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

# ----------------------------- Config ---------------------------------------
TICKER = "AAPL"
START, END = "2018-01-01", "2023-12-31"
TEST_DAYS = 60          # hold-out size in business days (~3 months)
LOOK_BACK = 60          # LSTM input window (days)
ARIMA_ORDER = (5, 1, 0) # (p, d, q), chosen manually
EPOCHS = 50             # upper limit, early stopping usually stops sooner
SEED = 42

np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)


# ----------------------------- Data -----------------------------------------
def load_data() -> pd.DataFrame:
    raw = yf.download(TICKER, start=START, end=END, auto_adjust=True, progress=False)
    # newer yfinance versions return MultiIndex columns like ("Close", "AAPL")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw[["Close"]].dropna()
    df.index = pd.to_datetime(df.index)
    # regular business-day index; US market holidays become NaN, so forward-fill
    df = df.asfreq("B").ffill()
    return df


# ----------------------------- Models ---------------------------------------
def run_naive(train: pd.DataFrame, n_steps: int) -> np.ndarray:
    """Baseline: tomorrow's price = last known price."""
    return np.full(n_steps, train["Close"].iloc[-1], dtype=float)


def run_arima(train: pd.DataFrame, n_steps: int) -> np.ndarray:
    fit = ARIMA(train["Close"], order=ARIMA_ORDER).fit()
    return np.asarray(fit.forecast(steps=n_steps))


def run_prophet(train: pd.DataFrame, test_index: pd.DatetimeIndex) -> np.ndarray:
    train_p = pd.DataFrame({"ds": train.index, "y": train["Close"].values})
    model = Prophet(daily_seasonality=False)
    model.fit(train_p)
    # predict exactly the test dates so every model is judged on the same days
    forecast = model.predict(pd.DataFrame({"ds": test_index}))
    return forecast["yhat"].values


def make_windows(arr: np.ndarray, look_back: int):
    X, y = [], []
    for i in range(look_back, len(arr)):
        X.append(arr[i - look_back:i, 0])
        y.append(arr[i, 0])
    return np.array(X).reshape(-1, look_back, 1), np.array(y)


def run_lstm(train: pd.DataFrame, n_steps: int) -> np.ndarray:
    # scaler is fitted on TRAIN ONLY (no leakage from the test period)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(train[["Close"]])

    X, y = make_windows(scaled, LOOK_BACK)

    model = Sequential([
        Input(shape=(LOOK_BACK, 1)),
        LSTM(50, return_sequences=True),
        LSTM(50),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(X, y, epochs=EPOCHS, batch_size=32, validation_split=0.1,
              callbacks=[early_stop], verbose=1)

    # recursive multi-step forecast: each prediction is fed back as input
    window = scaled[-LOOK_BACK:, 0].tolist()
    preds = []
    for _ in range(n_steps):
        x = np.array(window[-LOOK_BACK:]).reshape(1, LOOK_BACK, 1)
        p = float(model.predict(x, verbose=0)[0, 0])
        preds.append(p)
        window.append(p)
    return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).ravel()


# ----------------------------- Metrics --------------------------------------
def evaluate(name: str, actual, pred) -> dict:
    actual = np.asarray(actual, dtype=float).ravel()
    pred = np.asarray(pred, dtype=float).ravel()
    err = actual - pred
    return {
        "Model": name,
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAE": float(np.mean(np.abs(err))),
        "MAPE %": float(np.mean(np.abs(err / actual)) * 100),
    }


# ----------------------------- Plots ----------------------------------------
def draw_forecast(ax, train, test, pred, label, color, context=250):
    ax.plot(train.index[-context:], train["Close"].iloc[-context:],
            label=f"Train (last {context} days)", color="blue")
    ax.plot(test.index, test["Close"], label="Actual (test)", color="black")
    ax.plot(test.index, pred, label=label, color=color)
    ax.axvline(test.index[0], linestyle="--", color="gray")
    ax.set_title(f"{label} vs Actual ({TICKER}, {len(test)} business-day hold-out)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.grid(True)
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=45)


def draw_history(ax, train, test):
    ax.plot(train.index, train["Close"], label="Train", color="blue")
    ax.plot(test.index, test["Close"], label="Test (hold-out)", color="black")
    ax.axvline(test.index[0], linestyle="--", color="gray")
    ax.set_title(f"{TICKER} Closing Price ({START[:4]}-{END[:4]})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.grid(True)
    ax.legend()


def save_single(draw_fn, filename, *args):
    fig, ax = plt.subplots(figsize=(14, 6))
    draw_fn(ax, *args)
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


# ----------------------------- Main -----------------------------------------
def main():
    df = load_data()
    train, test = df.iloc[:-TEST_DAYS], df.iloc[-TEST_DAYS:]
    print(f"Total: {len(df)} | Train: {len(train)} | Test: {len(test)} business days")
    print(f"Test period: {test.index[0].date()} to {test.index[-1].date()}")

    # 1. Historical plot
    save_single(draw_history, "1_historical_plot.png", train, test)

    # 2-4. Fit on train, forecast the test dates
    preds = {
        "Naive (last value)": run_naive(train, len(test)),
        "ARIMA": run_arima(train, len(test)),
        "Prophet": run_prophet(train, test.index),
        "LSTM": run_lstm(train, len(test)),
    }

    save_single(draw_forecast, "2_arima_forecast.png",
                train, test, preds["ARIMA"], "ARIMA", "orange")
    save_single(draw_forecast, "3_prophet_forecast.png",
                train, test, preds["Prophet"], "Prophet", "green")
    save_single(draw_forecast, "4_lstm_forecast.png",
                train, test, preds["LSTM"], "LSTM", "red")

    # Combined figure
    fig, axs = plt.subplots(4, 1, figsize=(14, 20))
    draw_history(axs[0], train, test)
    draw_forecast(axs[1], train, test, preds["ARIMA"], "ARIMA", "orange")
    draw_forecast(axs[2], train, test, preds["Prophet"], "Prophet", "green")
    draw_forecast(axs[3], train, test, preds["LSTM"], "LSTM", "red")
    fig.tight_layout()
    fig.savefig("combined_forecast.png")
    plt.close(fig)

    # 5. Metrics
    results = pd.DataFrame([evaluate(n, test["Close"], p) for n, p in preds.items()])
    results = results.sort_values("RMSE").reset_index(drop=True)
    results.to_csv("metrics.csv", index=False)
    print("\nHold-out performance (lower is better):")
    print(results.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
