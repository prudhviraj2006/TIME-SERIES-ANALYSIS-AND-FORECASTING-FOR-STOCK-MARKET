import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from keras.models import Sequential
from keras.layers import LSTM, Dense

# --- Page Setup ---
st.set_page_config(page_title="📈 Stock Forecasting Dashboard", layout="centered")
st.title("📉 Stock Market Time Series Forecasting")

# --- Sidebar Input ---
st.sidebar.header("Select Parameters")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, INFY)", value='AAPL')
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-12-31"))
model_choice = st.sidebar.radio("Select Model", ["ARIMA", "Prophet", "LSTM"])

if st.sidebar.button("Run Forecast"):
    # --- Load Data ---
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        st.error("No data found for the given ticker.")
        st.stop()
    
    st.subheader(f"Raw Close Price for {ticker}")
    st.line_chart(data['Close'])

    close = data['Close'].dropna()

    # --- Forecasting ---
    if model_choice == "ARIMA":
        st.subheader("ARIMA Forecast")
        train = close[:int(0.8*len(close))]
        test = close[int(0.8*len(close)):]
        model = ARIMA(train, order=(5,1,0))
        fitted = model.fit()
        forecast = fitted.forecast(steps=len(test))
        rmse = np.sqrt(mean_squared_error(test, forecast))

        fig, ax = plt.subplots()
        ax.plot(test.index, test, label="Actual")
        ax.plot(test.index, forecast, label="Forecast")
        ax.set_title("ARIMA Prediction")
        ax.legend()
        st.pyplot(fig)
        st.write(f"📊 RMSE: {rmse:.4f}")

    elif model_choice == "Prophet":
        st.subheader("Prophet Forecast")
        df = data.reset_index()[['Date', 'Close']]
        df.columns = ['ds', 'y']
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=60)
        forecast = model.predict(future)
        fig = model.plot(forecast)
        st.pyplot(fig)

    elif model_choice == "LSTM":
        st.subheader("LSTM Forecast")
        # Scale data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(close.values.reshape(-1, 1))

        X, y = [], []
        window = 60
        for i in range(window, len(scaled_data)):
            X.append(scaled_data[i-window:i, 0])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

        predicted = model.predict(X_test)
        predicted = scaler.inverse_transform(predicted.reshape(-1, 1))
        actual = scaler.inverse_transform(y_test.reshape(-1, 1))
        rmse = np.sqrt(mean_squared_error(actual, predicted))

        fig, ax = plt.subplots()
        ax.plot(actual, label="Actual")
        ax.plot(predicted, label="Predicted")
        ax.set_title("LSTM Prediction")
        ax.legend()
        st.pyplot(fig)
        st.write(f"📊 RMSE: {rmse:.4f}")
