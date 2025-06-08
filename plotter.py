import matplotlib.pyplot as plt
import tempfile
import os

def plot_signal(df, signal):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'], label='Цена')
    plt.axvline(signal['timestamp'], color='orange', linestyle='--', alpha=0.5)
    plt.axhline(signal['entry'], color='green', linestyle='--', label='Entry')
    plt.axhline(signal['tp'], color='blue', linestyle='--', label='Take Profit')
    plt.axhline(signal['sl'], color='red', linestyle='--', label='Stop Loss')
    plt.title(f"Сигнал: {signal['direction']} на {df.name if hasattr(df, 'name') else 'активе'}")
    plt.legend()
    plt.grid(True)

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
        plt.savefig(tmpfile.name)
        plt.close()
        return tmpfile.name
