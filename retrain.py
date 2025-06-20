import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Permute, Multiply
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime
import os

DATA_FILE = "data/sp500_data.csv"
MODEL_DIR = "models"
LIVE_MODEL = os.path.join(MODEL_DIR, "lstm_model.h5")
LOG_FILE = os.path.join(MODEL_DIR, "model_log.csv")

# --- Feature Engineering ---
def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

def compute_indicators(df):
    df['MA'] = df.groupby('Ticker')['Price'].transform(lambda x: x.rolling(5).mean())
    df['STD'] = df.groupby('Ticker')['Price'].transform(lambda x: x.rolling(5).std())
    df['RSI'] = df.groupby('Ticker')['Price'].transform(lambda x: rsi(x))
    return df.dropna()

def prepare_sequences(df, steps=10):
    features = ['Price', 'Sentiment', 'MA', 'STD', 'RSI']
    X, y_class, y_reg = [], [], []

    for ticker in df['Ticker'].unique():
        sub = df[df['Ticker'] == ticker][features + ['Target_Class', 'Target_Return']].dropna().reset_index(drop=True)
        if len(sub) > steps:
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(sub[features])
            for i in range(steps, len(sub)):
                X.append(scaled[i-steps:i])
                y_class.append(sub['Target_Class'].iloc[i])
                y_reg.append(sub['Target_Return'].iloc[i])
    return np.array(X), np.array(y_class), np.array(y_reg)

# --- Model Architecture ---
def attention_block(inputs):
    a = Permute((2, 1))(inputs)
    a = Dense(inputs.shape[1], activation='softmax')(a)
    a = Permute((2, 1))(a)
    return Multiply()([inputs, a])

def build_model(input_shape):
    inp = Input(shape=input_shape)
    x = LSTM(64, return_sequences=True)(inp)
    x = Dropout(0.3)(x)
    x = attention_block(x)
    x = LSTM(32)(x)
    x = Dropout(0.3)(x)
    out_class = Dense(1, activation='sigmoid', name='class_output')(x)
    out_reg = Dense(1, activation='linear', name='reg_output')(x)
    return Model(inputs=inp, outputs=[out_class, out_reg])

# --- Retraining Procedure ---
def retrain():
    if not os.path.exists(DATA_FILE):
        print("❌ Data file not found.")
        return

    df = pd.read_csv(DATA_FILE).sort_values("Timestamp")
    df = compute_indicators(df)
    df['Target_Class'] = (df.groupby('Ticker')['Price'].diff().shift(-1) > 0).astype(int)
    df['Target_Return'] = df.groupby('Ticker')['Price'].pct_change().shift(-1)
    df = df.dropna()

    X, y_class, y_reg = prepare_sequences(df)

    if X.size == 0:
        print("❌ No training sequences generated.")
        return

    model = build_model(X.shape[1:])
    model.compile(
        optimizer='adam',
        loss={'class_output': 'binary_crossentropy', 'reg_output': 'mse'},
        metrics={'class_output': 'accuracy', 'reg_output': 'mae'}
    )

    print(f"🧠 Training on {X.shape[0]} sequences...")
    model.fit(
        X,
        {'class_output': y_class, 'reg_output': y_reg},
        batch_size=128,
        epochs=10,
        validation_split=0.2,
        callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
        verbose=1
    )

    # Save versioned model
    os.makedirs(MODEL_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    version_file = f"lstm_model_{timestamp}.h5"
    version_path = os.path.join(MODEL_DIR, version_file)

    model.save(version_path)
    model.save(LIVE_MODEL)

    # Log metadata
    log_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a") as f:
        if not log_exists:
            f.write("Version,Date,Train Size\n")
        f.write(f"{version_file},{datetime.now().isoformat()},{X.shape[0]}\n")

    print(f"✅ New model saved as: {version_file}")
    print(f"📌 Live model updated: {LIVE_MODEL}")

if __name__ == "__main__":
    retrain()