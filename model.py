from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
import numpy as np
import logging

SEQ_LENGTH = 20
THRESHOLD = 0.8
logging.basicConfig(level=logging.INFO)

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def walk_forward_train(X, y):
    tscv = TimeSeriesSplit(n_splits=5)
    best_model = None
    best_acc = 0

    X_seq = []
    y_seq = []

    for i in range(len(X) - SEQ_LENGTH):
        X_seq.append(X[i:i+SEQ_LENGTH])
        y_seq.append(1 if y[i+SEQ_LENGTH] > y[i] else 0)

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    for train_idx, test_idx in tscv.split(X_seq):
        X_train, X_test = X_seq[train_idx], X_seq[test_idx]
        y_train, y_test = y_seq[train_idx], y_seq[test_idx]

        model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
        model.fit(X_train, y_train, epochs=10, verbose=0)

        pred = (model.predict(X_test) > 0.5).astype(int)
        acc = accuracy_score(y_test, pred)

        if acc > best_acc:
            best_acc = acc
            best_model = model

    if best_acc < THRESHOLD:
        logging.info("Точность упала ниже порога. Модель требует переобучения.")

    return best_model, best_acc
