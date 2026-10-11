# 📈 Time Series Stock Forecasting (AAPL)

A data analytics project that forecasts Apple (AAPL) stock closing prices using three time series approaches: **ARIMA**, **Prophet**, and **LSTM**.

> 🔍 Built during my **Data Science & Analytics internship at Zidio Development** (Jun–Aug 2026, Team-15 project). It covers data collection, preprocessing, predictive modeling, and visualization on real stock market data.

---

## 📌 Project Objectives

- Understand time series components: **trend**, **seasonality**, **noise**
- Forecast stock prices using:
  - ARIMA (AutoRegressive Integrated Moving Average)
  - Prophet (Facebook's time series forecasting model)
  - LSTM (Long Short-Term Memory neural network)
- Visualize and compare the outputs of each model

---

## ⚙️ How It Works

1. Download AAPL daily closing prices (2018–2023) with `yfinance`
2. Set a business-day frequency and clean the series
3. **ARIMA(5,1,0):** forecast the next 30 business days
4. **Prophet:** fit trend + seasonality and forecast 30 days
5. **LSTM:** MinMax-scale prices, build 60-day windows, train two stacked LSTM layers (50 units each)
6. Save each plot plus a combined comparison figure

---

## 🛠 Tech Stack & Tools

| Tool / Library      | Purpose                             |
|---------------------|-------------------------------------|
| Python              | Core programming language           |
| yfinance            | Fetching historical stock data      |
| Pandas, NumPy       | Data processing & numerical ops     |
| Matplotlib          | Data visualization                  |
| Statsmodels         | ARIMA modeling                      |
| Prophet             | Trend + seasonality forecasting     |
| scikit-learn        | MinMax scaling for LSTM input       |
| TensorFlow/Keras    | Deep learning (LSTM)                |
| Git, GitHub         | Version control                     |

---

## 📂 Folder Structure

