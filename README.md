# Stock Trend Prediction Web App

A Flask-based web application that predicts and visualizes stock price trends for any company using deep learning (LSTM) and interactive charts.

---

## 🚀 Features

- **Predicts stock closing prices** using LSTM neural networks (Keras/TensorFlow)
- **Dynamic ticker support:** Enter any valid stock ticker (e.g., `AAPL`, `POWERGRID.NS`)
- **Automatic data fetching** from Yahoo Finance (with robust CSV fallback)
- **Interactive charts:** 
  - Closing Price vs Time with 20/50 and 100/200 day EMAs
  - Model Prediction vs Actual Trend
- **Descriptive statistics** table for the selected stock
- **Downloadable CSV** of the processed dataset
- **Modern, responsive UI** with Bootstrap 5
- **Clear error handling** for missing or malformed data

---

## 🖥️ Screenshots

![App Screenshot](static/ema_20_50.png)
*Example: Closing Price vs Time (20 & 50 Days EMA)*

---

## ⚙️ How It Works

1. **User enters a stock ticker** (e.g., `AAPL`) in the web form.
2. The app **downloads historical data** from Yahoo Finance (or loads from a local CSV if rate-limited).
3. **Data is cleaned and processed**; moving averages and descriptive stats are calculated.
4. An **LSTM model is trained** on the stock’s historical data and predicts future trends.
5. **Charts and tables are displayed** in the browser, and the dataset can be downloaded as CSV.

---

## 🛠️ Tech Stack

- **Backend:** Python, Flask
- **Deep Learning:** Keras, TensorFlow
- **Data:** Pandas, yfinance, scikit-learn
- **Visualization:** Matplotlib, Bootstrap 5
- **Frontend:** HTML, CSS, Bootstrap

---

## 📦 Setup Instructions

1. **Clone the repository:**
    ```bash
    git clone https://github.com/jallella01/stock-trend-prediction.git
    cd stock-trend-prediction
    ```

2. **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    .venv\Scripts\activate  # On Windows
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the app:**
    ```bash
    python app.py
    ```
    The app will be available at [http://127.0.0.1:5000](http://127.0.0.1:5000).

---

## 📁 Project Structure

```
PredictStockTrend(FLASK)/
│
├── app.py
├── requirements.txt
├── static/
│   ├── ema_20_50.png
│   ├── ema_100_200.png
│   ├── stock_prediction.png
│   └── [stock]_dataset.csv
├── templates/
│   └── index.html
└── ...
```

---

## 📝 Notes

- If Yahoo Finance rate-limits you, place a valid CSV for the ticker in the `static/` folder as `[TICKER]_dataset.csv`.
- CSVs must have columns: `Date,Open,High,Low,Close,Adj Close,Volume`.
- The app is for educational/demo purposes and not for real financial advice.

---

## 📧 Contact

For questions or suggestions, open an issue or contact [jallellakarthik@gmail.com](mailto:jallellakarthik@gmail.com).

---

**Enjoy predicting stock trends!**
