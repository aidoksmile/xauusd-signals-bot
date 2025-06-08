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
LOOKBACK_DAYS = 730  # История за 2 года
TRADE_HORIZON = 3  # Прогноз на 3 дня вперёд
MODEL_PATH = "xauusd_model.pkl"
GRAPH_PATH = "xauusd_signal.png"

# === Функции работы с данными ===
def fetch_data(ticker, lookback_days):
    try:
        logging.info("Начало загрузки данных...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        data = yf.download(ticker, start=start_date, end=end_date, interval='1d')
        logging.info(f"Полученные столбцы: {data.columns.tolist()}")
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
📈 [XAU/USD Signal] {signal}

💰 Current Price: {current_price:.2f}
🔹 Entry: {entry:.2f}
🔸 TP: {tp:.2f}
🔻 SL: {sl:.2f}
🪙 Risk: {risk:.2f}%
📊 Accuracy: {accuracy * 100:.2f}%
🕒 Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}
        """
        await bot.send_message(chat_id=CHAT_ID, text=message)

        # Генерация графика
        generate_graph(current_price, entry, tp, sl)
        with open(GRAPH_PATH, 'rb') as photo:
            await bot.send_photo(chat_id=CHAT_ID, photo=photo, caption="📊 График с сигналом")
        logging.info(f"Telegram signal sent: {signal}, Цена: {current_price}, Точность: {accuracy:.2f}")
    except Exception as e:
        logging.error(f"Ошибка отправки Telegram-сообщения: {str(e)}")
        raise

# === Расчёт входа и выхода ===
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

# === Генерация графика с сигналом ===
def generate_graph(current_price, entry, tp, sl):
    try:
        df_plot = fetch_data(TICKER, LOOKBACK_DAYS)
        plt.figure(figsize=(12, 6))
        plt.plot(df_plot.index, df_plot['Close'], label='Цена XAU/USD', color='black', alpha=0.5)
        plt.axhline(entry, color='blue', linestyle='--', label='Entry')
        plt.axhline(tp, color='green', linestyle='--', label='Take Profit')
        plt.axhline(sl, color='red', linestyle='--', label='Stop Loss')
        plt.title('XAU/USD с торговыми уровнями')
        plt.legend()
        plt.savefig(GRAPH_PATH)
        plt.close()
        logging.info("График успешно создан")
    except Exception as e:
        logging.error(f"Ошибка создания графика: {str(e)}")
        raise

# === Основная логика бота ===
def main():
    try:
        logging.info("=== НАЧАЛО ВЫПОЛНЕНИЯ MAIN ===")
        df = fetch_data(TICKER, LOOKBACK_DAYS)

        X, y, full_data = prepare_features(df, TRADE_HORIZON)
        model, accuracy = train_or_load_model(X, y)

        last_row = X.iloc[-1].values.reshape(1, -1)
        prediction = model.predict(last_row)[0]
        current_price = float(df['Close'].iloc[-1])

        signal = "BUY" if prediction == 1 else "SELL"
        entry, tp, sl, risk = calculate_entry_tp_sl(current_price, signal)

        asyncio.run(send_telegram_signal(signal, entry, tp, sl, risk, current_price, accuracy))
        logging.info(f"Сигнал отправлен: {signal}, Entry: {entry}, TP: {tp}, SL: {sl}, Цена: {current_price}")

    except Exception as e:
        logging.error(f"КРИТИЧЕСКАЯ ОШИБКА: {str(e)}", exc_info=True)

# === Flask App ===
app = Flask(__name__)

@app.route('/')
def index():
    main()  # Выполняем основную логику
    return jsonify({"status": "OK", "message": "Signal sent successfully"})

@app.route('/signal')
def manual_signal():
    main()  # Можно вызвать вручную через /signal
    return jsonify({"status": "OK", "message": "Signal manually triggered!"})

@app.errorhandler(Exception)
def handle_exception(e):
    logging.exception("Uncaught exception: %s", str(e))
    return jsonify(error=str(e)), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
