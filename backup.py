import streamlit as st
import pandas as pd
import numpy as np
import os
import math
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM

# Function to get user's choice of stock
def select_stock():
    stocks = ["ADANIPORTS", "TCS", "ASIANPAINT", "AXISBANK", "BAJAJ-AUTO", "BAJAJFINSV", "BAJFINANCE", "BHARTIARTL",
              "BPCL", "BRITANNIA", "CIPLA", "COALINDIA", "DRREDDY", "EICHERMOT", "GAIL", "GRASIM", "HCLTECH", "HDFC",
              "HDFCBANK", "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK", "INDUSINDBK", "INFRATEL", "INFY", "IOC",
              "ITC", "JSWSTEEL", "KOTAKBANK", "LT"]

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
def build_train_model(train_data, epochs=10, batch_size=32, selected_stock=""):
    training_status = st.empty()
    training_status.info("Checking for pre-trained model...")

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(train_data['Close'].values.reshape(-1, 1))

    train_data_len = len(scaled_data)

    x_train, y_train = [], []
    for i in range(60, train_data_len):
        x_train.append(scaled_data[i-60:i, 0])
        y_train.append(scaled_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Check if the model has already been trained and saved
    model_filename = f'models/{selected_stock}_prediction_model.h5'
    if os.path.exists(model_filename):
        # Load the pre-trained model
        model = load_model(model_filename)
        training_status.success("Pre-trained model loaded successfully!")
    else:
        # Build and train the model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dense(units=25))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

        training_status.success("Model trained successfully!")

        # Save the trained model
        model.save(model_filename)  # Assuming a folder named 'models'

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

    # Create a DataFrame with predictions and dates
    display_with_predictions = pd.DataFrame(predictions, index=prediction_dates, columns=['Predictions'])

    # Example buy/sell logic (you may need to adjust this based on your strategy)
    display_with_predictions['Buy_Signal'] = np.where(display_with_predictions['Predictions'] > display_with_predictions['Predictions'].shift(1), 1, 0)
    display_with_predictions['Sell_Signal'] = np.where(display_with_predictions['Predictions'] < display_with_predictions['Predictions'].shift(1), 1, 0)

    return prediction_dates, display_with_predictions

# Function to display results with signals
def display_results_with_signals(train, display, predictions):
    st.title("Stock Prediction Results")

    # Line chart for training data
    st.line_chart(train['Close'], use_container_width=True)
    st.text("Training Data: Actual Stock Prices")

    # Line chart for test data and predictions
    display_with_predictions = display.copy()
    display_with_predictions['Predictions'] = np.nan  # Initialize 'Predictions' column with NaN values
    display_with_predictions['Predictions'].iloc[-len(predictions):] = predictions['Predictions'].values.flatten()

    # Ensure predictions and display_with_predictions have the same length
    if len(predictions) > len(display_with_predictions):
        predictions = predictions[:len(display_with_predictions)]
    elif len(predictions) < len(display_with_predictions):
        display_with_predictions = display_with_predictions.iloc[:len(predictions)]

    # Convert predictions to a DataFrame to use the shift operation
    predictions_df = pd.DataFrame(predictions['Predictions'].values, index=display_with_predictions.index, columns=['Predictions'])

    # Example buy/sell logic (you may need to adjust this based on your strategy)
    display_with_predictions['Buy_Signal'] = np.where(predictions_df['Predictions'] > predictions_df['Predictions'].shift(1), 1, 0)
    display_with_predictions['Sell_Signal'] = np.where(predictions_df['Predictions'] < predictions_df['Predictions'].shift(1), 1, 0)

    st.line_chart(display_with_predictions[['Close', 'Predictions']], use_container_width=True)
    st.text("Test Data: Actual vs Predicted Stock Prices with Buy/Sell Signals")

    # Get the 5 most recent buy and sell signals
    recent_buy_signals = display_with_predictions[display_with_predictions['Buy_Signal'] == 1].index[-5:]
    recent_sell_signals = display_with_predictions[display_with_predictions['Sell_Signal'] == 1].index[-5:]

    # Create adjacent columns for buy and sell signals
    col1, col2 = st.columns(2)

    # Display recent buy signals in a column with green color
    with col1:
        st.subheader("Recent Buy Signals:")
        for buy_signal in recent_buy_signals:
            st.markdown(f"**{buy_signal.strftime('%Y-%m-%d')}**", unsafe_allow_html=True)
            st.markdown('<font color="green">▲ Buy</font>', unsafe_allow_html=True)

    # Display recent sell signals in a column with red color
    with col2:
        st.subheader("Recent Sell Signals:")
        for sell_signal in recent_sell_signals:
            st.markdown(f"**{sell_signal.strftime('%Y-%m-%d')}**", unsafe_allow_html=True)
            st.markdown('<font color="red">▼ Sell</font>', unsafe_allow_html=True)


# Main Streamlit app with buy/sell functionality
def main():
    st.set_page_config(page_title="Stock Prediction App", layout="wide")

    selected_stock = select_stock()
    df = load_stock_data(selected_stock)
    train_data = df[:math.ceil(len(df) * 0.8)]
    test_data = df[math.ceil(len(df) * 0.8) - 60:]

    # Allow user customization of epochs and batch size
    epochs = st.sidebar.slider("Select the number of epochs", min_value=1, max_value=20, value=10)
    batch_size = st.sidebar.slider("Select the batch size", min_value=1, max_value=64, value=32)

    model, scaler = build_train_model(train_data, epochs, batch_size, selected_stock)

    # Get prediction dates and prices
    prediction_dates, predictions = make_predictions_with_dates(model, scaler, test_data)

    train = df[:len(train_data)]
    display = pd.DataFrame(index=prediction_dates)
    display['Predictions'] = np.nan
    display['Predictions'].iloc[-len(predictions):] = predictions['Predictions'].values

    # Display the results with buy/sell signals
    # Function to display results with signals
def display_results_with_signals(train, display, predictions):
    st.title("Stock Prediction Results with Buy/Sell Signals")

    # Line chart for training data
    st.line_chart(train['Close'], use_container_width=True)
    st.text("Training Data: Actual Stock Prices")

    # Line chart for test data and predictions
    display_with_signals = display.copy()
    display_with_signals['Buy_Signal'] = predictions['Buy_Signal'].values
    display_with_signals['Sell_Signal'] = predictions['Sell_Signal'].values

    st.line_chart(display_with_signals[['Close', 'Buy_Signal', 'Sell_Signal']], use_container_width=True)
    st.text("Test Data: Actual Stock Prices with Buy/Sell Signals")

    # Display the predicted prices along with dates
    st.title("Predicted Stock Prices with Buy/Sell Signals")

    # Display the predicted prices along with dates
    st.line_chart(display_with_signals[['Predictions', 'Buy_Signal', 'Sell_Signal']], use_container_width=True)
    st.text("Predicted Stock Prices with Buy/Sell Signals")

    # Display the predicted price for a specific date
    st.header("Predicted Price for a Specific Date")

    # Use the minimum date as the default date
    default_date = predictions.index.min().to_pydatetime().date()

    selected_date = st.date_input("Select a date", min_value=predictions.index.min(), max_value=predictions.index.max(), value=default_date)

    selected_date_str = selected_date.strftime("%Y-%m-%d")

    try:
        predicted_price = display.loc[selected_date_str, 'Predictions']
        buy_signal = predictions.loc[selected_date_str, 'Buy_Signal']
        sell_signal = predictions.loc[selected_date_str, 'Sell_Signal']

        # Display the predicted price with buy/sell signal
        st.write(f"The predicted stock price for {selected_date_str} is: {predicted_price:.2f}")

        if buy_signal == 1:
            st.success("Buy Signal!")
        elif sell_signal == 1:
            st.error("Sell Signal!")
        else:
            st.warning("No signal for the selected date.")

    except KeyError:
        st.warning("No prediction available for the selected date. Please choose another date.")

if __name__ == "__main__":
    main()
