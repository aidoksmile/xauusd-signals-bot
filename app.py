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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from joblib import dump, load

# === Нейросеть (LSTM) ===
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# === Telegram Bot ===
import telegram
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
    CallbackContext
)

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
MODEL_PATH = "xauusd_lstm_model.pkl"
GRAPH_PATH = "xauusd_signal.png"
HISTORY_CSV = "trades_history.csv"

CHECK_INTERVAL = 900  # Проверка каждые 15 минут
MIN_ACCURACY_THRESHOLD = 0.6  # Если точность ниже — переобучаем модель

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
    delta = series.diff().dropna()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def prepare_features(data, horizon=3):
    try:
        logging.info("Подготовка признаков для LSTM...")
        data['Target'] = (data['Close'].shift(-horizon).pct_change(horizon) > 0).astype(int)
        data.dropna(inplace=True)

        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        X = data[features].values
        y = data['Target'].values

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        X_seq = []
        for i in range(60, len(X_scaled)):
            X_seq.append(X_scaled[i-60:i, :])
        X_seq = np.array(X_seq)
        y = y[60:]
        logging.info("Признаки подготовлены")
        return X_seq, y, data.iloc[60:]
    except Exception as e:
        logging.error(f"Ошибка подготовки признаков: {str(e)}")
        raise

# === Модель LSTM ===
def train_or_load_model(X_train, y_train):
    try:
        logging.info("Попытка загрузить модель...")
        model = load(MODEL_PATH)
        logging.info("Модель загружена")
        return model
    except FileNotFoundError:
        logging.info("Модель не найдена. Обучение новой LSTM-модели...")

        model = Sequential()
        model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(0.2))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=20, batch_size=32, callbacks=[early_stop], verbose=0)
        dump(model, MODEL_PATH)
        logging.info("Модель LSTM обучена и сохранена")
        return model

# === Отправка сигнала в Telegram ===
async def send_telegram_signal(signal, entry, tp, sl, risk, current_price):
    try:
        bot = telegram.Bot(token=TELEGRAM_TOKEN)

        message = f"""
📈 [XAU/USD Signal] *{signal}*

💰 Current Price: *{current_price:.2f}*
🔹 Entry: *{entry:.2f}*
🔸 TP: *{tp:.2f}*
🔻 SL: *{sl:.2f}*
🪙 Risk: *{risk:.2f}%*
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

# === Walk-forward обучение ===
def walk_forward_training(X, y):
    tscv = TimeSeriesSplit(n_splits=5)
    best_model = None
    best_accuracy = 0.0

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = Sequential()
        model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(0.2))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=20, batch_size=32, callbacks=[early_stop], verbose=0)

        pred = model.predict(X_test, verbose=0)
        accuracy = accuracy_score(y_test, (pred > 0.5).flatten())
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

    dump(best_model, MODEL_PATH)
    logging.info(f"Лучшая модель обновлена. Точность: {best_accuracy:.2f}")
    return best_model, best_accuracy

# === Backtesting ===
def run_backtest(model, df, X):
    try:
        pred = model.predict(X, verbose=0)
        df = df.iloc[60:].copy()
        df['Signal'] = (pred > 0.5).flatten().astype(int)
        df['Signal'] = df['Signal'].map({True: 1, False: -1})

        df['Return'] = df['Close'].pct_change(TRADE_HORIZON).shift(-TRADE_HORIZON)
        df['Strategy_Return'] = df['Signal'].shift(1) * df['Return']

        cum_return = (df['Strategy_Return'] + 1).cumprod()
        buy_and_hold = (df['Return'] + 1).cumprod()

        plt.figure(figsize=(12, 6))
        plt.plot(cum_return, label='Стратегия')
        plt.plot(buy_and_hold, label='Buy & Hold')
        plt.title('Backtesting стратегии')
        plt.legend()
        plt.grid(True)
        plt.savefig("backtest.png")
        plt.close()

        final_strategy = cum_return.iloc[-1]
        final_bh = buy_and_hold.iloc[-1]
        win_rate = (df['Strategy_Return'] > 0).sum() / len(df[df['Strategy_Return'] != 0])
        profit_factor = df[df['Strategy_Return'] > 0]['Strategy_Return'].sum() / abs(df[df['Strategy_Return'] < 0]['Strategy_Return'].sum())

        logging.info(f"Backtesting: Strategy Return: {final_strategy:.2f}, Buy&Hold: {final_bh:.2f}, Win Rate: {win_rate:.2f}, Profit Factor: {profit_factor:.2f}")

        return {
            'strategy_return': final_strategy,
            'buy_and_hold': final_bh,
            'win_rate': win_rate,
            'profit_factor': profit_factor
        }
    except Exception as e:
        logging.error(f"Ошибка бэктестинга: {str(e)}")
        return {}

# === Сохранение истории сделок ===
def log_trade(signal, entry, tp, sl, current_price, accuracy, backtest_results):
    try:
        trade_log = pd.DataFrame([{
            'Date': datetime.now(),
            'Signal': signal,
            'Entry': entry,
            'TP': tp,
            'SL': sl,
            'Current Price': current_price,
            'Risk': abs(entry - sl) / current_price * 100,
            'Accuracy': accuracy,
            'Win Rate': backtest_results.get('win_rate', 0),
            'Profit Factor': backtest_results.get('profit_factor', 0)
        }])
        trade_log.to_csv(HISTORY_CSV, mode='a', header=not os.path.exists(HISTORY_CSV), index=False)
        logging.info("Сделка записана в историю")
    except Exception as e:
        logging.error(f"Ошибка записи сделки: {str(e)}")

# === Основная логика бота ===
def main():
    try:
        logging.info("=== НАЧАЛО ВЫПОЛНЕНИЯ MAIN ===")

        df_daily = fetch_data(TICKER, LOOKBACK_DAYS, interval='1d')
        X, y, full_data = prepare_features(df_daily, TRADE_HORIZON)

        model, accuracy = walk_forward_training(X, y)

        df_15m = fetch_data(TICKER, 7, interval='15m')
        current_price = float(df_15m['Close'].iloc[-1])

        prediction = model.predict(X[-1:], verbose=0)[0][0] > 0.5
        signal_str = "BUY" if prediction else "SELL"

        is_confirmed = check_short_term_confirmation(df_15m, signal_str)
        if not is_confirmed:
            logging.info(f"Сигнал {signal_str} не подтверждён → пропуск")
            return

        entry, tp, sl, risk = calculate_entry_tp_sl(current_price, signal_str)
        asyncio.run(send_telegram_signal(signal_str, entry, tp, sl, risk, current_price))
        backtest_results = run_backtest(model, full_data, X)
        log_trade(signal_str, entry, tp, sl, current_price, accuracy, backtest_results)

    except Exception as e:
        logging.error(f"КРИТИЧЕСКАЯ ОШИБКА: {str(e)}", exc_info=True)

# === Команды Telegram-бота ===
async def handle_start(update, context):
    await update.message.reply_text("🤖 Бот активен. Используйте:\n/signal - получить прогноз\n/history - история сделок\n/graph - график торговли\n/accuracy - точность модели")

async def handle_signal(update, context):
    df_daily = fetch_data(TICKER, LOOKBACK_DAYS, interval='1d')
    X, _, _ = prepare_features(df_daily, TRADE_HORIZON)
    model = load(MODEL_PATH)
    prediction = model.predict(X[-1:], verbose=0)[0][0] > 0.5
    signal_str = "BUY" if prediction else "SELL"
    await update.message.reply_text(f"📊 Текущий сигнал: *{signal_str}*", parse_mode="Markdown")

async def handle_history(update, context):
    if os.path.exists(HISTORY_CSV):
        history_df = pd.read_csv(HISTORY_CSV)
        await update.message.reply_text("📜 Последние 10 сделок:")
        for _, row in history_df.tail(10).iterrows():
            await update.message.reply_text(
                f"{row['Date']} | {row['Signal']} | "
                f"TP: {row['TP']:.2f} | SL: {row['SL']:.2f}"
            )
    else:
        await update.message.reply_text("❌ История сделок пуста")

async def handle_graph(update, context):
    if os.path.exists("backtest.png"):
        with open("backtest.png", 'rb') as photo:
            await update.message.reply_photo(photo, caption="📈 Equity Curve")
    else:
        await update.message.reply_text("❌ График ещё не доступен")

async def handle_accuracy(update, context):
    model = load(MODEL_PATH)
    df_daily = fetch_data(TICKER, LOOKBACK_DAYS, interval='1d')
    X, y, _ = prepare_features(df_daily, TRADE_HORIZON)
    pred = model.predict(X, verbose=0)
    acc = accuracy_score(y, (pred > 0.5).flatten())
    await update.message.reply_text(f"📊 Точность модели: *{acc * 100:.2f}%*", parse_mode="Markdown")

async def handle_unknown(update, context):
    await update.message.reply_text("❌ Неизвестная команда")

async def telegram_bot():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    
    app.add_handler(CommandHandler("start", handle_start))
    app.add_handler(CommandHandler("signal", handle_signal))
    app.add_handler(CommandHandler("history", handle_history))
    app.add_handler(CommandHandler("graph", handle_graph))
    app.add_handler(CommandHandler("accuracy", handle_accuracy))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_unknown))

    await app.run_polling()

# === Flask App ===
app = Flask(__name__)

@app.route('/')
def index():
    if request.method == 'GET':
        main()
    return {"status": "OK", "message": "Signal sent successfully"}

@app.route('/signal')
def manual_signal():
    main()
    return {"status": "OK", "message": "Signal manually triggered!"}

@app.errorhandler(Exception)
def handle_exception(e):
    logging.exception("Uncaught exception: %s", str(e))
    return {"error": str(e)}, 500

# === Цикл проверки ===
def run_continuously():
    while True:
        try:
            main()
        except Exception as e:
            logging.error(f"Ошибка в основном цикле: {str(e)}")
        time.sleep(CHECK_INTERVAL)

# === Запуск сервиса ===
if __name__ == "__main__":
    tele_thread = Thread(target=lambda: asyncio.run(telegram_bot()), daemon=True)
    tele_thread.start()

    checker_thread = Thread(target=run_continuously, daemon=True)
    checker_thread.start()

    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
