import time
import pandas as pd
import os
import logging
from data_loader import fetch_data
from model import walk_forward_train
from signal_processor import generate_signal
from bot_handler import send_signal
from backtester import backtest_strategy
from threading import Thread
from app import app as flask_app

os.environ.setdefault("CONFIG", ".env")
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)

def background_check():
    while True:
        try:
            df_day, df_15m = fetch_data()
            X = df_day[['Open', 'High', 'Low', 'Close', 'Volume']].values
            y = df_day['Close'].values
            model, acc = walk_forward_train(X, y)
            signal = generate_signal(model, df_day, df_15m)
            send_signal(signal)
        except Exception as e:
            logging.error(f"Ошибка: {e}")
        time.sleep(900)

if __name__ == "__main__":
    # Для Render: используем Gunicorn или запускаем Flask + фоновый поток
    from waitress import serve
    flask_thread = Thread(target=background_check)
    flask_thread.start()
    serve(flask_app, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
