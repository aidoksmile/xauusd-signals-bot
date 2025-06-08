import numpy as np
import pandas as pd

def generate_signal(model, df_daily, df_15min):
    X = df_daily[['Open', 'High', 'Low', 'Close', 'Volume']].values
    X_norm = (X - X.mean()) / X.std()

    X_input = X_norm[-SEQ_LENGTH:]
    prediction = model.predict(X_input.reshape(1, SEQ_LENGTH, X_input.shape[1]))[0][0]

    direction = 'BUY' if prediction > 0.5 else 'SELL'
    entry = df_15min['Close'].iloc[-1]
    tp = entry + 5 * (df_15min['Close'].pct_change().std())
    sl = entry - 3 * (df_15min['Close'].pct_change().std())
    risk = abs(entry - sl)

    return {
        'direction': direction,
        'entry': entry,
        'tp': tp,
        'sl': sl,
        'risk': risk,
        'accuracy': float(prediction),
        'timestamp': df_15min.index[-1],
        'df_15min': df_15min
    }
