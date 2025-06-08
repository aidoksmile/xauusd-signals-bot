import os
from flask import Flask, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump, load
import asyncio
import telegram

# === Настройки из переменных окружения ===
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
TICKER = 'GC=F'
LOOKBACK_DAYS = 730
TRADE_HORIZON = 3
MODEL_PATH = "xauusd_model.pkl"
LOG_FILE = "xauusd_bot.log"

# === Логирование ===
logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_data(ticker, lookback_days):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    data = yf.download(ticker, start=start_date, end=end_date, interval='1d')
    return data

def prepare_features(data, horizon):
    data['Return'] = data['Adj Close'].pct_change(horizon).shift(-horizon)
    data['Target'] = np.where(data['Return'] > 0, 1, -1)
    data['SMA_10'] = data['Adj Close'].rolling(10).mean()
    data['SMA_30'] = data['Adj Close'].rolling(30).mean()
    data['RSI'] = compute_rsi(data['Adj Close'], 14)
    data.dropna(inplace=True)
    X = data[['SMA_10', 'SMA_30', 'RSI']]
    y = data['Target']
    return X, y, data

def compute_rsi(series, window):
    delta = series.diff().dropna()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def train_or_load_model(X, y):
    try:
        model = load(MODEL_PATH)
        logging.info("Модель загружена")
    except FileNotFoundError:
        model = RandomForestClassifier(n_estimators=100)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        logging.info(f"Model Accuracy: {accuracy_score(y_test, pred):.2f}")
        dump(model, MODEL_PATH)
    return model

async def send_telegram_signal(signal, entry, tp, sl, risk):
    bot = telegram.Bot(token=TELEGRAM_TOKEN)
    message = f"""
📈 [XAU/USD Signal] {signal}

🔹 Entry: {entry:.2f}
🔸 TP: {tp:.2f}
🔻 SL: {sl:.2f}
🪙 Risk: {risk:.2f}%
🕒 Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}
    """
    await bot.send_message(chat_id=CHAT_ID, text=message)
    logging.info(f"Telegram signal sent: {signal}")

def calculate_entry_tp_sl(price, direction):
    volatility = price * 0.01
    if direction == "BUY":
        entry = price
        tp = entry + 2 * volatility
        sl = entry - volatility
    else:
        entry = price
        tp = entry - 2 * volatility
        sl = entry + volatility
    risk = abs(entry - sl) / entry * 100 * 0.02 * 100
    return entry, tp, sl, risk

def main():
    df = fetch_data(TICKER, LOOKBACK_DAYS)
    X, y, full_data = prepare_features(df, TRADE_HORIZON)
    model = train_or_load_model(X, y)

    last_row = X.iloc[-1].values.reshape(1, -1)
    prediction = model.predict(last_row)[0]
    current_price = df['Adj Close'].iloc[-1]

    signal = "BUY" if prediction == 1 else "SELL"
    entry, tp, sl, risk = calculate_entry_tp_sl(current_price, signal)
    asyncio.run(send_telegram_signal(signal, entry, tp, sl, risk))
    logging.info(f"Signal sent: {signal}, Entry: {entry}, TP: {tp}, SL: {sl}")

# === Flask App ===
app = Flask(__name__)

@app.route('/')
def index():
    main()  # Выполняем основную логику
    return jsonify({"status": "OK", "message": "Signal sent successfully"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
