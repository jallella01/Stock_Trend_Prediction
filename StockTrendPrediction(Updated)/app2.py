import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from flask import Flask, render_template, request, send_file

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from twelvedata import TDClient
from dotenv import load_dotenv

from datetime import date

import os

# -----------------------------------
# Flask App
# -----------------------------------

app = Flask(__name__)

plt.style.use("fivethirtyeight")

# -----------------------------------
# Load API Key
# -----------------------------------

load_dotenv()

api_key = os.getenv("TWELVEDATA_API_KEY")

td = TDClient(apikey=api_key)

# -----------------------------------
# Date Range
# -----------------------------------

START = "2000-01-01"

TODAY = date.today().strftime("%Y-%m-%d")

# -----------------------------------
# Load Stock Data
# -----------------------------------

def load_data(ticker):

    try:

        ticker = ticker.upper().strip()

        ts = td.time_series(
            symbol=ticker,
            interval="1day",
            start_date=START,
            end_date=TODAY,
            outputsize=5000
        )

        data = ts.as_pandas()

        if data is None or data.empty:

            return None

        data.reset_index(inplace=True)

        data.rename(columns={
            'datetime': 'Date',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }, inplace=True)

        numeric_cols = [
            'Open',
            'High',
            'Low',
            'Close',
            'Volume'
        ]

        for col in numeric_cols:

            if col in data.columns:

                data[col] = pd.to_numeric(
                    data[col],
                    errors='coerce'
                )

        data['Date'] = pd.to_datetime(
            data['Date'],
            errors='coerce'
        )

        data = data.dropna(subset=['Date'])

        data = data.sort_values(
            by='Date',
            ascending=True
        )

        data.reset_index(
            drop=True,
            inplace=True
        )

        return data

    except Exception as e:

        print(str(e))

        return None


# -----------------------------------
# Home Route
# -----------------------------------

@app.route('/', methods=['GET', 'POST'])

def index():

    if request.method == 'POST':

        stock = request.form.get('stock')

        if not stock or stock.strip() == "":

            return render_template(
                'index.html',
                error="Please enter a stock ticker."
            )

        stock = stock.upper().strip()

        # -----------------------------------
        # Load Data
        # -----------------------------------

        data = load_data(stock)

        if data is None:

            return render_template(
                'index.html',
                error=f"'{stock}' is not supported in free tier version. Please try another ticker."
            )

        # -----------------------------------
        # Descriptive Statistics
        # -----------------------------------

        data_desc = data.describe()

        # -----------------------------------
        # EMA Calculations
        # -----------------------------------

        ema20 = data.Close.ewm(
            span=20,
            adjust=False
        ).mean()

        ema50 = data.Close.ewm(
            span=50,
            adjust=False
        ).mean()

        ema100 = data.Close.ewm(
            span=100,
            adjust=False
        ).mean()

        ema200 = data.Close.ewm(
            span=200,
            adjust=False
        ).mean()

        # -----------------------------------
        # Train / Validation / Test Split
        # -----------------------------------

        total_data = len(data)

        train_size = int(total_data * 0.70)

        validation_size = int(total_data * 0.15)

        train = pd.DataFrame(
            data[0:train_size]
        )

        validation = pd.DataFrame(
            data[
                train_size:
                train_size + validation_size
            ]
        )

        test = pd.DataFrame(
            data[
                train_size + validation_size:
            ]
        )

        # -----------------------------------
        # Scaling
        # -----------------------------------

        scaler = MinMaxScaler(
            feature_range=(0,1)
        )

        train_close = train[['Close']].values

        validation_close = validation[['Close']].values

        test_close = test[['Close']].values

        data_training_array = scaler.fit_transform(
            train_close
        )

        data_validation_array = scaler.transform(
            validation_close
        )

        data_testing_array = scaler.transform(
            test_close
        )

        # -----------------------------------
        # Create Training Sequences
        # -----------------------------------

        x_train = []
        y_train = []

        for i in range(
            100,
            data_training_array.shape[0]
        ):

            x_train.append(
                data_training_array[i-100:i]
            )

            y_train.append(
                data_training_array[i, 0]
            )

        x_train = np.array(x_train)

        y_train = np.array(y_train)

        # -----------------------------------
        # Create Validation Sequences
        # -----------------------------------

        x_val = []
        y_val = []

        for i in range(
            100,
            data_validation_array.shape[0]
        ):

            x_val.append(
                data_validation_array[i-100:i]
            )

            y_val.append(
                data_validation_array[i, 0]
            )

        x_val = np.array(x_val)

        y_val = np.array(y_val)

        # -----------------------------------
        # Build LSTM Model
        # -----------------------------------

        model = Sequential()

        model.add(
            LSTM(
                units=50,
                activation='tanh',
                input_shape=(
                    x_train.shape[1],
                    1
                )
            )
        )

        model.add(
            Dropout(0.2)
        )

        model.add(
            Dense(1)
        )

        # -----------------------------------
        # Compile Model
        # -----------------------------------

        model.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=['mean_absolute_error']
        )

        # -----------------------------------
        # Early Stopping
        # -----------------------------------

        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=2,
            restore_best_weights=True
        )

        # -----------------------------------
        # Train Model
        # -----------------------------------

        model.fit(

            x_train,
            y_train,

            validation_data=(
                x_val,
                y_val
            ),

            epochs=20,

            batch_size=32,

            callbacks=[early_stop],

            verbose=0
        )

        # -----------------------------------
        # Prepare Test Data
        # -----------------------------------

        past_100_days = pd.DataFrame(
            validation_close[-100:]
        )

        test_df = pd.DataFrame(
            test_close
        )

        final_df = pd.concat(
            [past_100_days, test_df],
            ignore_index=True
        )

        input_data = scaler.transform(
            final_df
        )

        # -----------------------------------
        # Create Test Sequences
        # -----------------------------------

        x_test = []
        y_test = []

        for i in range(
            100,
            input_data.shape[0]
        ):

            x_test.append(
                input_data[i-100:i]
            )

            y_test.append(
                input_data[i, 0]
            )

        x_test = np.array(x_test)

        y_test = np.array(y_test)

        # -----------------------------------
        # Predictions
        # -----------------------------------

        y_pred = model.predict(
            x_test,
            verbose=0
        )

        y_pred = scaler.inverse_transform(
            y_pred
        )

        y_test_actual = scaler.inverse_transform(
            y_test.reshape(-1,1)
        )

        # -----------------------------------
        # Create Static Folder
        # -----------------------------------

        if not os.path.exists("static"):

            os.makedirs("static")

        # -----------------------------------
        # Plot 1
        # -----------------------------------

        fig1, ax1 = plt.subplots(figsize=(12,6))

        ax1.plot(
            data['Date'],
            data['Close'],
            label='Close Price'
        )

        ax1.plot(
            data['Date'],
            ema20,
            label='EMA 20'
        )

        ax1.plot(
            data['Date'],
            ema50,
            label='EMA 50'
        )

        ax1.set_title(
            f"{stock} Closing Price with EMA 20 & 50"
        )

        ax1.set_xlabel("Date")

        ax1.set_ylabel("Price")

        ax1.legend()

        ax1.grid(True)

        ema_chart_path = "static/ema_20_50.png"

        fig1.savefig(ema_chart_path)

        plt.close(fig1)

        # -----------------------------------
        # Plot 2
        # -----------------------------------

        fig2, ax2 = plt.subplots(figsize=(12,6))

        ax2.plot(
            data['Date'],
            data['Close'],
            label='Close Price'
        )

        ax2.plot(
            data['Date'],
            ema100,
            label='EMA 100'
        )

        ax2.plot(
            data['Date'],
            ema200,
            label='EMA 200'
        )

        ax2.set_title(
            f"{stock} Closing Price with EMA 100 & 200"
        )

        ax2.set_xlabel("Date")

        ax2.set_ylabel("Price")

        ax2.legend()

        ax2.grid(True)

        ema_chart_path_100_200 = "static/ema_100_200.png"

        fig2.savefig(ema_chart_path_100_200)

        plt.close(fig2)

        # -----------------------------------
        # Plot 3
        # -----------------------------------

        fig3, ax3 = plt.subplots(figsize=(12,6))

        ax3.plot(
            y_test_actual,
            'g',
            label='Actual Price'
        )

        ax3.plot(
            y_pred,
            'r',
            label='Predicted Price'
        )

        ax3.set_title(
            f"{stock} Predicted vs Actual"
        )

        ax3.set_xlabel("Time")

        ax3.set_ylabel("Price")

        ax3.legend()

        ax3.grid(True)

        prediction_chart_path = "static/stock_prediction.png"

        fig3.savefig(prediction_chart_path)

        plt.close(fig3)

        # -----------------------------------
        # Save Dataset
        # -----------------------------------

        dataset_filename = f"{stock}_dataset.csv"

        data.to_csv(
            f"static/{dataset_filename}",
            index=False
        )

        # -----------------------------------
        # Save Model
        # -----------------------------------

        model.save("keras_model.h5")

        # -----------------------------------
        # Render HTML
        # -----------------------------------

        return render_template(

            'index.html',

            stock_name=stock,

            plot_path_ema_20_50=ema_chart_path,

            plot_path_ema_100_200=ema_chart_path_100_200,

            plot_path_prediction=prediction_chart_path,

            data_desc=data_desc.to_html(
                classes='table table-bordered'
            ),

            dataset_link=f"static/{dataset_filename}"
        )

    return render_template('index.html')


# -----------------------------------
# Download Route
# -----------------------------------

@app.route('/download/<filename>')

def download_file(filename):

    return send_file(
        f"static/{filename}",
        as_attachment=True
    )


# -----------------------------------
# Run Flask App
# -----------------------------------

if __name__ == '__main__':

    app.run(debug=True)