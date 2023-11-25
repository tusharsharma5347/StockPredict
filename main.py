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
        "ADANIPORTS","TCS", "ASIANPAINT", "AXISBANK", "BAJAJ-AUTO", "BAJAJFINSV", "BAJFINANCE", "BHARTIARTL", "BPCL", "BRITANNIA", "CIPLA", "COALINDIA", "DRREDDY", "EICHERMOT", "GAIL", "GRASIM", "HCLTECH", "HDFC", "HDFCBANK", "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK", "INDUSINDBK", "INFRATEL", "INFY", "IOC", "ITC", "JSWSTEEL", "KOTAKBANK", "LT"]

    st.sidebar.title("Stock Prediction App")
    selected_stock = st.sidebar.selectbox("Select a stock", stocks)
    return selected_stock

# Function to load and preprocess stock data
def load_stock_data(selected_stock):
    df = pd.read_csv(f'Data/{selected_stock}.csv')  # Assuming your data files are in a folder named 'Data'
    df['Date'] = pd.to_datetime(df['Date'])  # Convert 'Date' column to datetime format
    df.set_index('Date', inplace=True)  # Set 'Date' as the index
    return df

# Function to build and train the LSTM model
def build_train_model(train_data, epochs=10, batch_size=32):
    st.info("Training the model... Please wait.")
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
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

    st.success("Model trained successfully!")

    return model, scaler

# Function to make predictions with dates
def make_predictions_with_dates(model, scaler, test_data):
    test_data_len = len(test_data)

    inputs = scaler.transform(test_data['Close'].values.reshape(-1, 1))

    x_test = []
    for i in range(60, test_data_len):
        x_test.append(inputs[i-60:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Get corresponding dates
    prediction_dates = test_data.index[60:]

    return prediction_dates, predictions

# Function to display results
def display_results(train, display, predictions):
    st.title("Stock Prediction Results")

    # Line chart for training data
    st.line_chart(train['Close'], use_container_width=True)
    st.text("Training Data: Actual Stock Prices")

    # Line chart for test data and predictions
    display_with_predictions = display.copy()
    display_with_predictions['Predictions'] = np.nan  # Initialize 'Predictions' column with NaN values
    display_with_predictions['Predictions'].iloc[-len(predictions):] = predictions.flatten()

    st.line_chart(display_with_predictions[['Close', 'Predictions']], use_container_width=True)
    st.text("Test Data: Actual vs Predicted Stock Prices")

# Main Streamlit app
def main():
    st.set_page_config(page_title="Stock Prediction App", layout="wide")

    selected_stock = select_stock()
    df = load_stock_data(selected_stock)
    train_data = df[:math.ceil(len(df) * 0.8)]
    test_data = df[math.ceil(len(df) * 0.8) - 60:]

    # Allow user customization of epochs and batch size
    epochs = st.sidebar.slider("Select the number of epochs", min_value=1, max_value=20, value=10)
    batch_size = st.sidebar.slider("Select the batch size", min_value=1, max_value=64, value=32)

    model, scaler = build_train_model(train_data, epochs, batch_size)

    # Get prediction dates and prices
    prediction_dates, predictions = make_predictions_with_dates(model, scaler, test_data)

    train = df[:len(train_data)]
    display = df[len(train_data)-60:]
    display['Predictions'] = np.nan  # Initialize 'Predictions' column with NaN values
    display['Predictions'].iloc[-len(predictions):] = predictions.flatten()

    # Debugging print statements
    print("Prediction Dates:", prediction_dates)
    print("Predictions:", predictions)

    # Display the predicted prices along with dates
    display_results(train, display, predictions)

    # Display the predicted price for a specific date
    st.header("Predicted Price for a Specific Date")

    # Use the minimum date as the default date
    default_date = prediction_dates.min().to_pydatetime().date()

    selected_date = st.date_input("Select a date", min_value=prediction_dates.min(), max_value=prediction_dates.max(), value=default_date)

    selected_date_str = selected_date.strftime("%Y-%m-%d")

    try:
        predicted_price = display.loc[selected_date_str, 'Predictions']
        st.write(f"The predicted stock price for {selected_date_str} is: {predicted_price:.2f}")
    except KeyError:
        st.warning("No prediction available for the selected date. Please choose another date.")

if __name__ == "__main__":
    main()
