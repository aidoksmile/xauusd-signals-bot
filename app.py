from flask import Flask, jsonify
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Здесь будет временный кэш последнего сигнала
latest_signal = None

@app.route('/')
def home():
    return "XAU/USD Trade Bot is running"

@app.route('/signal')
def get_signal():
    if latest_signal:
        return jsonify(latest_signal)
    else:
        return jsonify({"error": "No signal available yet"})
