import os
import sys
import logging
import asyncio
from datetime import datetime, timedelta

# === Flask ===
from flask import Flask, jsonify

# === ML / Data ===
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load

# === Telegram Bot ===
import telegram

# === Настройка логирования ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# === Переменные окружения ===
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

TICKER = 'GC=F'  # XAU/USD
LOOKBACK_DAYS = 730  # История за 2 года
TRADE_HORIZON = 3  # Прогноз на 3 дня вперёд
MODEL_PATH = "xauusd_model.pkl"

# === Функции работы с данными ===
def fetch_data(ticker, lookback_days):
    try:
        logging.info("Начало загрузки данных...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        data = yf.download(ticker, start=start_date, end=end_date, interval='1d')
        logging.info("Данные успешно загружены")
        return data
    except Exception as e:
        logging.error(f"Ошибка загрузки данных: {str(e)}")
        raise

def compute_rsi(series, window=14):
    try:
        delta = series.diff().dropna()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    except Exception as e:
        logging.error(f"Ошибка вычисления RSI: {str(e)}")
        raise

def prepare_features(data, horizon):
    try:
        logging.info("Подготовка признаков...")
        data['Return'] = data['Adj Close'].pct_change(horizon).shift(-horizon)
        data['Target'] = np.where(data['Return'] > 0, 1, -1)  # BUY: 1, SELL: -1
        data['SMA_10'] = data['Adj Close'].rolling(10).mean()
        data['SMA_30'] = data['Adj Close'].rolling(30).mean()
        data['RSI'] = compute_rsi(data['Adj Close'], 14)
        data.dropna(inplace=True)
        X = data[['SMA_10', 'SMA_30', 'RSI']]
        y = data['Target']
        logging.info("Признаки подготовлены")
        return X, y, data
    except Exception as e:
        logging.error(f"Ошибка подготовки признаков: {str(e)}")
        raise

# === Модель ML ===
def train_or_load_model(X, y):
    try:
        logging.info("Попытка загрузить модель...")
        model = load(MODEL_PATH)
        logging.info("Модель загружена")
        return model
    except FileNotFoundError:
        logging.info("Модель не найдена. Обучение новой модели...")
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X, y)
        dump(model, MODEL_PATH)
        logging.info("Модель обучена и сохранена")
        return model

# === Отправка сигнала в Telegram ===
async def send_telegram_signal(signal, entry, tp, sl, risk):
    try:
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
    except Exception as e:
        logging.error(f"Ошибка отправки Telegram-сообщения: {str(e)}")
        raise

# === Расчёт входа и выхода ===
def calculate_entry_tp_sl(price, direction):
    try:
        volatility = price * 0.01
        if direction == "BUY":
            entry = price
            tp = entry + 2 * volatility
            sl = entry - volatility
        else:
            entry = price
            tp = entry - 2 * volatility
            sl = entry + volatility
        risk = abs(entry - sl) / entry * 100 * 0.02 * 100  # 2% от депозита
        return entry, tp, sl, risk
    except Exception as e:
        logging.error(f"Ошибка расчёта входа/выхода: {str(e)}")
        raise

# === Основная логика бота ===
def main():
    try:
        logging.info("=== НАЧАЛО ВЫПОЛНЕНИЯ MAIN ===")
        df = fetch_data(TICKER, LOOKBACK_DAYS)
        X, y, full_data = prepare_features(df, TRADE_HORIZON)
        model = train_or_load_model(X, y)

        last_row = X.iloc[-1].values.reshape(1, -1)
        prediction = model.predict(last_row)[0]
        current_price = df['Adj Close'].iloc[-1]

        signal = "BUY" if prediction == 1 else "SELL"
        entry, tp, sl, risk = calculate_entry_tp_sl(current_price, signal)
        asyncio.run(send_telegram_signal(signal, entry, tp, sl, risk))
        logging.info(f"Сигнал отправлен: {signal}, Entry: {entry}, TP: {tp}, SL: {sl}")

    except Exception as e:
        logging.error(f"КРИТИЧЕСКАЯ ОШИБКА: {str(e)}", exc_info=True)

# === Flask App ===
app = Flask(__name__)

@app.route('/')
def index():
    main()  # Выполняем основную логику
    return jsonify({"status": "OK", "message": "Signal sent successfully"})

# === Глобальный обработчик ошибок ===
@app.errorhandler(Exception)
def handle_exception(e):
    logging.exception("Uncaught exception: %s", str(e))
    return jsonify(error=str(e)), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
