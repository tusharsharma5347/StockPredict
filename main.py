import streamlit as st
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM


# Function to get user's choice of stock
def select_stock():
    stocks = [
        "TCS", "MSFT", "GOOGL", "AMZN", "FB",
        "TSLA", "NVDA", "INTC", "AMD", "CSCO",
        "IBM", "ORCL", "QCOM", "V", "PYPL",
        "NFLX", "DIS", "GS", "JPM", "BAC",
        "WMT", "AMGN", "PFE", "JNJ", "MRK",
        "BA", "CAT", "XOM", "CVX", "T",
        "VZ", "C", "GS", "IBM", "INTC",
        "MCD", "KO", "PEP", "JPM", "WFC",
        "PG", "UNH", "MMM", "HD", "AXP",
        "AAPL", "GS", "CSCO", "MSFT", "DIS"
    ]

    st.sidebar.title("Stock Prediction App")
    selected_stock = st.sidebar.selectbox("Select a stock", stocks)
    return selected_stock

# Function to load and preprocess stock data
def load_stock_data(selected_stock):
    df = pd.read_csv(f'/Users/tushar/Desktop/StockPredict/Data/{selected_stock}.csv')  # Assuming your data files are named after the stock symbols
    return df

# Function to build and train the LSTM model
def build_train_model(train_data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(train_data['Close'].values.reshape(-1, 1))

    train_data_len = len(scaled_data)

    x_train, y_train = [], []
    for i in range(60, train_data_len):
        x_train.append(scaled_data[i-60:i, 0])
        y_train.append(scaled_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=10, batch_size=32)

    return model, scaler

# Function to make predictions
def make_predictions(model, scaler, test_data):
    test_data_len = len(test_data)

    inputs = scaler.transform(test_data['Close'].values.reshape(-1, 1))

    x_test = []
    for i in range(60, test_data_len):
        x_test.append(inputs[i-60:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    return predictions

# Function to display results
def display_results(train, display, predictions):
    st.title("Stock Prediction Results")
    st.line_chart(train['Close'], use_container_width=True)
    st.line_chart(display[['Close', 'Predictions']], use_container_width=True)

# Main Streamlit app
def main():
    st.set_page_config(page_title="Stock Prediction App", layout="wide")

    selected_stock = select_stock()
    df = load_stock_data(selected_stock)
    train_data = df[:math.ceil(len(df) * 0.8)]
    test_data = df[math.ceil(len(df) * 0.8) - 60:]

    model, scaler = build_train_model(train_data)
    predictions = make_predictions(model, scaler, test_data)

    train = df[:len(train_data)]
    display = df[len(train_data)-60:]
    display['Predictions'] = predictions[:len(display)]



    display_results(train, display, predictions)

if __name__ == "__main__":
    main()
