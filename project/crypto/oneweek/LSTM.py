import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Load Microsoft stock data
symbols_list = ["BTC-USD",'ETH-USD','BNB-USD','SOL-USD','XRP-USD','DOGE-USD','SBIN.BO']



TODAY= date.today().strftime("%Y-%m-%d")
START = pd.to_datetime(TODAY)-pd.DateOffset(365*8)

ticker= st.selectbox("Select company for prediction ", symbols_list)
# Download the data
data = yf.download(ticker, START, TODAY)
def close(data):
    data = data[['Close']]

# Normalize data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

# Create sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 7
X, y = create_sequences(data_scaled, seq_length)

# Split data into train and test sets
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape X_train for LSTM input (samples, timesteps, features)
X_train = X_train.reshape(X_train.shape[0], seq_length, 1)

# Build LSTM model
model = Sequential()
model.add(LSTM(units=64, return_sequences=False, input_shape=(seq_length, 1)))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=10)

# Predict stock prices for the next week
future_days = 7
future_dates = pd.date_range(start=data.index[-1], periods=future_days + 1)[1:]
future_prices = []

for i in range(future_days):
    last_seq = X[-1]  # Get the last sequence
    last_seq = last_seq.reshape(1, seq_length, 1)  # Reshape for prediction

    next_price = model.predict(last_seq)
    future_prices.append(next_price[0][0])

    # Update X for next prediction (append predicted price and remove oldest)
    X = np.append(X, last_seq, axis=0)  # Append predicted sequence
    X = X[-seq_length:]                 # Keep only the last 'seq_length' sequences

# Inverse transform to get actual prices
future_prices = scaler.inverse_transform(np.array(future_prices).reshape(-1, 1))

# Create Streamlit app
st.title(ticker+" Price Prediction")
st.write("Predicting prices for the next week using LSTM model")

st.line_chart(data['Close'])
st.write("Predicted prices for the next week:")
st.line_chart(pd.DataFrame(future_prices, index=future_dates, columns=['Predicted Price']))

def sopen(data):
    data = data[['Open']]
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

# Create sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
    return np.array(X), np.array(y)

    seq_length = 7
    X, y = create_sequences(data_scaled, seq_length)

# Split data into train and test sets
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape X_train for LSTM input (samples, timesteps, features)
X_train = X_train.reshape(X_train.shape[0], seq_length, 1)

# Build LSTM model
model = Sequential()
model.add(LSTM(units=64, return_sequences=False, input_shape=(seq_length, 1)))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=10)

# Predict stock prices for the next week
future_days = 7
future_dates = pd.date_range(start=data.index[-1], periods=future_days + 1)[1:]
future_prices = []

for i in range(future_days):
    last_seq = X[-1]  # Get the last sequence
    last_seq = last_seq.reshape(1, seq_length, 1)  # Reshape for prediction

    next_price = model.predict(last_seq)
    future_prices.append(next_price[0][0])

    # Update X for next prediction (append predicted price and remove oldest)
    X = np.append(X, last_seq, axis=0)  # Append predicted sequence
    X = X[-seq_length:]                 # Keep only the last 'seq_length' sequences

# Inverse transform to get actual prices
future_prices = scaler.inverse_transform(np.array(future_prices).reshape(-1, 1))

# Create Streamlit app
st.title(ticker+" Price Prediction")
st.write("Predicting prices for the next week using LSTM model")

st.line_chart(data['Open'])
st.write("Predicted prices for the next week:")
st.line_chart(pd.DataFrame(future_prices, index=future_dates, columns=['Predicted Price']))
def high(data):
    data = data[['High']]

# Normalize data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Create sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 7
X, y = create_sequences(data_scaled, seq_length)

# Split data into train and test sets
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape X_train for LSTM input (samples, timesteps, features)
X_train = X_train.reshape(X_train.shape[0], seq_length, 1)

# Build LSTM model
model = Sequential()
model.add(LSTM(units=64, return_sequences=False, input_shape=(seq_length, 1)))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=10)

# Predict stock prices for the next week
future_days = 7
future_dates = pd.date_range(start=data.index[-1], periods=future_days + 1)[1:]
future_prices = []

for i in range(future_days):
    last_seq = X[-1]  # Get the last sequence
    last_seq = last_seq.reshape(1, seq_length, 1)  # Reshape for prediction

    next_price = model.predict(last_seq)
    future_prices.append(next_price[0][0])

    # Update X for next prediction (append predicted price and remove oldest)
    X = np.append(X, last_seq, axis=0)  # Append predicted sequence
    X = X[-seq_length:]                 # Keep only the last 'seq_length' sequences

# Inverse transform to get actual prices
future_prices = scaler.inverse_transform(np.array(future_prices).reshape(-1, 1))

# Create Streamlit app
st.title(ticker+" Price Prediction")
st.write("Predicting prices for the next week using LSTM model")

st.line_chart(data['High'])
st.write("Predicted prices for the next week:")
st.line_chart(pd.DataFrame(future_prices, index=future_dates, columns=['Predicted Price']))
def  low(data):
    data = data[['Low']]

# Normalize data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

# Create sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 7
X, y = create_sequences(data_scaled, seq_length)

# Split data into train and test sets
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape X_train for LSTM input (samples, timesteps, features)
X_train = X_train.reshape(X_train.shape[0], seq_length, 1)

# Build LSTM model
model = Sequential()
model.add(LSTM(units=64, return_sequences=False, input_shape=(seq_length, 1)))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=10)

# Predict stock prices for the next week
future_days = 7
future_dates = pd.date_range(start=data.index[-1], periods=future_days + 1)[1:]
future_prices = []

for i in range(future_days):
    last_seq = X[-1]  # Get the last sequence
    last_seq = last_seq.reshape(1, seq_length, 1)  # Reshape for prediction

    next_price = model.predict(last_seq)
    future_prices.append(next_price[0][0])

    # Update X for next prediction (append predicted price and remove oldest)
    X = np.append(X, last_seq, axis=0)  # Append predicted sequence
    X = X[-seq_length:]                 # Keep only the last 'seq_length' sequences

# Inverse transform to get actual prices
future_prices = scaler.inverse_transform(np.array(future_prices).reshape(-1, 1))

# Create Streamlit app
st.title(ticker+" Price Prediction")
st.write("Predicting prices for the next week using LSTM model")

st.line_chart(data['Low'])
st.write("Predicted prices for the next week:")
st.line_chart(pd.DataFrame(future_prices, index=future_dates, columns=['Predicted Price']))

Methods=("close","open","high","low")
selected_method= st.selectbox("Select dataset for prediction ", Methods)
if selected_method == "close":
    close(data)
if selected_method == "open":
    sopen(data)
if selected_method == "high":
    high(data)
if selected_method == "low":
    low(data)

