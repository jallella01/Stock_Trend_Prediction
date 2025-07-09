import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from flask import Flask, render_template, request, send_file
import datetime as dt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import os
plt.style.use("fivethirtyeight")

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        stock = request.form.get('stock')
        if not stock:
            stock = 'POWERGRID.NS'  # Default stock if none is entered

        # Define the start and end dates for stock data
        start = dt.datetime(2000, 1, 1)
        end = dt.datetime(2024, 10, 1)

        # Download stock data
        try:
            df = yf.download(stock, start=start, end=end)
            if df.empty or df.shape[0] == 0:
                raise Exception("No data from Yahoo")
        except Exception as e:
            # Try to load from CSV if available
            csv_path = f"static/{stock}_dataset.csv"
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    df.columns = df.columns.str.strip()
                    if 'Date' not in df.columns or 'Close' not in df.columns:
                        return render_template(
                            'index.html',
                            error=f"CSV for '{stock}' must have 'Date' and 'Close' columns. Please upload a valid stock CSV."
                        )
                    df = df.dropna(subset=['Date'])
                    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                    df = df.dropna(subset=['Date'])
                    df.set_index('Date', inplace=True)
                except Exception as e:
                    return render_template(
                        'index.html',
                        error=f"Error reading CSV for '{stock}': {str(e)}"
                    )
            else:
                return render_template(
                    'index.html',
                    error=f"No data found for '{stock}' and Yahoo Finance is rate-limiting you. Please try again later or upload a CSV."
                )

        # Descriptive Data
        data_desc = df.describe()

        # Exponential Moving Averages
        ema20 = df.Close.ewm(span=20, adjust=False).mean()
        ema50 = df.Close.ewm(span=50, adjust=False).mean()
        ema100 = df.Close.ewm(span=100, adjust=False).mean()
        ema200 = df.Close.ewm(span=200, adjust=False).mean()

        # Data splitting
        data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
        data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

        # Scaling data
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_training_array = scaler.fit_transform(data_training)

        # Prepare training data for LSTM
        x_train, y_train = [], []
        for i in range(100, data_training_array.shape[0]):
            x_train.append(data_training_array[i-100:i])
            y_train.append(data_training_array[i, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)

        # Build and train the model
        model = Sequential()
        model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=60, activation='relu', return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(units=80, activation='relu', return_sequences=True))
        model.add(Dropout(0.4))
        model.add(LSTM(units=120, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)  # Reduce epochs for faster response

        # Prepare test data
        past_100_days = data_training.tail(100)
        final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
        input_data = scaler.transform(final_df)

        x_test, y_test = [], []
        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i - 100:i])
            y_test.append(input_data[i, 0])
        x_test, y_test = np.array(x_test), np.array(y_test)

        # Make predictions
        y_predicted = model.predict(x_test)
        y_predicted = scaler.inverse_transform(y_predicted)
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Ensure static directory exists
        if not os.path.exists("static"):
            os.makedirs("static")

        # Plot 1: Closing Price vs Time Chart with 20 & 50 Days EMA
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(df.Close, 'y', label='Closing Price')
        ax1.plot(ema20, 'g', label='EMA 20')
        ax1.plot(ema50, 'r', label='EMA 50')
        ax1.set_title("Closing Price vs Time (20 & 50 Days EMA)")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Price")
        ax1.legend()
        ema_chart_path = "static/ema_20_50.png"
        fig1.savefig(ema_chart_path)
        plt.close(fig1)

        # Plot 2: Closing Price vs Time Chart with 100 & 200 Days EMA
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(df.Close, 'y', label='Closing Price')
        ax2.plot(ema100, 'g', label='EMA 100')
        ax2.plot(ema200, 'r', label='EMA 200')
        ax2.set_title("Closing Price vs Time (100 & 200 Days EMA)")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Price")
        ax2.legend()
        ema_chart_path_100_200 = "static/ema_100_200.png"
        fig2.savefig(ema_chart_path_100_200)
        plt.close(fig2)

        # Plot 3: Prediction vs Original Trend
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        ax3.plot(y_test, 'g', label="Original Price", linewidth=1)
        ax3.plot(y_predicted, 'r', label="Predicted Price", linewidth=1)
        ax3.set_title("Prediction vs Original Trend")
        ax3.set_xlabel("Time")
        ax3.set_ylabel("Price")
        ax3.legend()
        prediction_chart_path = "static/stock_prediction.png"
        fig3.savefig(prediction_chart_path)
        plt.close(fig3)

        # Save dataset as CSV
        df.to_csv("static/POWERGRID.NS_dataset.csv", index=False)

        # Return the rendered template with charts and dataset
        return render_template('index.html',
                               plot_path_ema_20_50=ema_chart_path,
                               plot_path_ema_100_200=ema_chart_path_100_200,
                               plot_path_prediction=prediction_chart_path,
                               data_desc=data_desc.to_html(classes='table table-bordered'),
                               dataset_link="static/POWERGRID.NS_dataset.csv")

    return render_template('index.html')

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(f"static/{filename}", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
