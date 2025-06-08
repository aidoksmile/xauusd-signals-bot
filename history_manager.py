import csv
import os
import time

CSV_PATH = "trades.csv"

def init_history():
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "direction", "entry", "tp", "sl", "risk", "accuracy"])

def save_trade(signal):
    with open(CSV_PATH, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            signal['timestamp'],
            signal['direction'],
            signal['entry'],
            signal['tp'],
            signal['sl'],
            signal['risk'],
            signal['accuracy']
        ])
