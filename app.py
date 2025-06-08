import os
import sys
import logging
import asyncio
from datetime import datetime, timedelta

# === Flask и фоновый запуск ===
from flask import Flask, request
from threading import Thread
import time

# === ML / Data ===
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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
LOOKBACK_DAYS = 730  # История обучения (2 года)
TRADE_HORIZON = 3  # Прогноз на 3 дня вперёд
MODEL_PATH = "xauusd_model.pkl"
GRAPH_PATH = "xauusd_signal.png"

CHECK_INTERVAL = 900  # 15 минут в секундах

# === Функции работы с данными ===
def fetch_data(ticker, lookback_days, interval='1d'):
    try:
        logging.info(f"Начало загрузки данных ({interval})...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)

        last_price = float(data['Close'].iloc[-1])
        logging.info(f"Данные успешно загружены: {data.shape[0]} свечей | Последняя цена: {last_price:.2f}")
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
        logging.info("Подготовка признаков для модели...")
        data['Return'] = data['Close'].pct_change(horizon).shift(-horizon)
        data['Target'] = np.where(data['Return'] > 0, 1, -1)  # BUY: 1, SELL: -1
        data['SMA_10'] = data['Close'].rolling(10).mean()
        data['SMA_30'] = data['Close'].rolling(30).mean()
        data['RSI'] = compute_rsi(data['Close'], 14)
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
        pred = model.predict(X)
        acc = accuracy_score(y, pred)
        logging.info(f"Модель загружена. Точность: {acc:.2f}")
        return model, acc
    except FileNotFoundError:
        logging.info("Модель не найдена. Обучение новой модели...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        dump(model, MODEL_PATH)
        logging.info(f"Модель обучена и сохранена. Точность: {acc:.2f}")
        return model, acc

# === Отправка сигнала в Telegram ===
async def send_telegram_signal(signal, entry, tp, sl, risk, current_price, accuracy):
    try:
        bot = telegram.Bot(token=TELEGRAM_TOKEN)

        message = f"""
📈 [XAU/USD Signal] *{signal}*

💰 Current Price: *{current_price:.2f}*
🔹 Entry: *{entry:.2f}*
🔸 TP: *{tp:.2f}*
🔻 SL: *{sl:.2f}*
🪙 Risk: *{risk:.2f}%*
📊 Accuracy: *{accuracy * 100:.2f}%*
🕒 Time: *{datetime.now().strftime('%Y-%m-%d %H:%M')}*
        """

        generate_graph(current_price, entry, tp, sl)
        with open(GRAPH_PATH, 'rb') as photo:
            await bot.send_photo(chat_id=CHAT_ID, photo=photo, caption=message, parse_mode='Markdown')

        logging.info(f"Telegram signal sent: {signal}, Цена: {current_price:.2f}")
    except Exception as e:
        logging.error(f"Ошибка отправки Telegram-сообщения: {str(e)}")
        raise

# === Расчёт уровней ===
def calculate_entry_tp_sl(price, direction):
    try:
        price = float(price)
        volatility = price * 0.01
        if direction == "BUY":
            entry = price
            tp = entry + 2 * volatility
            sl = entry - volatility
        else:
            entry = price
            tp = entry - 2 * volatility
            sl = entry + volatility
        risk = abs(entry - sl) / price * 100 * 0.02 * 100  # 2% от депозита
        return entry, tp, sl, risk
    except Exception as e:
        logging.error(f"Ошибка расчёта входа/выхода: {str(e)}")
        raise

# === Генерация графика ===
def generate_graph(current_price, entry, tp, sl):
    try:
        df_plot = fetch_data(TICKER, 7, interval='15m')
        plt.figure(figsize=(12, 6))
        plt.plot(df_plot.index, df_plot['Close'], label='Цена XAU/USD', color='black', alpha=0.7)
        plt.axhline(entry, color='blue', linestyle='--', label='Entry')
        plt.axhline(tp, color='green', linestyle='--', label='Take Profit')
        plt.axhline(sl, color='red', linestyle='--', label='Stop Loss')
        plt.title('XAU/USD | Текущая неделя (15min)')
        plt.legend()
        plt.grid(True)
        plt.savefig(GRAPH_PATH)
        plt.close()
        logging.info("График успешно создан")
    except Exception as e:
        logging.error(f"Ошибка создания графика: {str(e)}")
        raise

# === Анализ младшего ТФ для подтверждения сигнала ===
def check_short_term_confirmation(df_15m, signal):
    try:
        df_15m['RSI'] = compute_rsi(df_15m['Close'], window=14)
        latest_price = float(df_15m['Close'].iloc[-1])
        avg_price = float(df_15m['Close'].iloc[-5:].mean())
        rsi = float(df_15m['RSI'].iloc[-1])

        trend = 'up' if latest_price > avg_price else 'down'

        if signal == "BUY" and trend == "up" and rsi < 60:
            return True
        elif signal == "SELL" and trend == "down" and rsi > 40:
            return True
        return False
    except Exception as e:
        logging.error(f"Ошибка анализа младшего ТФ: {str(e)}")
        return False

# === Основная логика бота ===
def main():
    try:
        logging.info("=== НАЧАЛО ВЫПОЛНЕНИЯ MAIN ===")

        # Загрузка дневных данных для модели
        df_daily = fetch_data(TICKER, LOOKBACK_DAYS, interval='1d')
        X, y, _ = prepare_features(df_daily, TRADE_HORIZON)
        model, accuracy = train_or_load_model(X, y)

        # Получаем последнюю цену из дневных данных
        daily_price = float(df_daily['Close'].iloc[-1])

        # Прогноз по дневной модели
        last_row = X.iloc[-1].values.reshape(1, -1)
        prediction = model.predict(last_row)[0]
        signal_str = "BUY" if prediction == 1 else "SELL"

        # Загружаем 15-минутные данные для текущей цены и анализа
        df_15m = fetch_data(TICKER, 7, interval='15m')
        current_price = float(df_15m['Close'].iloc[-1])

        # Проверяем сигнал на младшем ТФ
        is_confirmed = check_short_term_confirmation(df_15m, signal_str)
        if not is_confirmed:
            logging.info(f"Сигнал {signal_str} не подтверждён → пропуск")
            return

        # Если всё подтверждено — рассчитываем уровни и отправляем
        entry, tp, sl, risk = calculate_entry_tp_sl(current_price, signal_str)
        asyncio.run(send_telegram_signal(signal_str, entry, tp, sl, risk, current_price, accuracy))
        logging.info(f"Сигнал отправлен: {signal_str}, Entry: {entry:.2f}, TP: {tp:.2f}, SL: {sl:.2f}")

    except Exception as e:
        logging.error(f"КРИТИЧЕСКАЯ ОШИБКА: {str(e)}", exc_info=True)

# === Цикл проверки ===
def run_continuously():
    while True:
        try:
            main()
        except Exception as e:
            logging.error(f"Ошибка в основном цикле: {str(e)}")
        time.sleep(CHECK_INTERVAL)  # 15 минут

# === Flask App ===
app = Flask(__name__)

@app.route('/')
def index():
    return {"status": "OK", "message": "Signal sent successfully"}

@app.route('/signal')
def manual_signal():
    main()
    return {"status": "OK", "message": "Signal manually triggered!"}

@app.errorhandler(Exception)
def handle_exception(e):
    logging.exception("Uncaught exception: %s", str(e))
    return {"error": str(e)}, 500

# === Запуск фонового потока ===
if __name__ == "__main__":
    thread = Thread(target=run_continuously)
    thread.daemon = True
    thread.start()

    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
