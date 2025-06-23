import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Permute, Multiply, Lambda, Concatenate, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K

# === Config ===
DATA_FILE = "data/sp500_data.csv"
MODEL_FILE = "models/lstm_model.keras"
METRICS_FILE = "models/metrics.json"
SEQUENCE_STEPS = 10
SEED = 42

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / (avg_loss + 1e-6)
    return 100 - (100 / (1 + rs))

def compute_indicators(df):
    df = df.copy()
    df["MA"] = df.groupby("Ticker")["Price"].transform(lambda x: x.rolling(5).mean())
    df["STD"] = df.groupby("Ticker")["Price"].transform(lambda x: x.rolling(5).std())
    df["RSI"] = df.groupby("Ticker")["Price"].transform(lambda x: rsi(x))
    df["Momentum"] = df.groupby("Ticker")["Price"].transform(lambda x: x.diff(3))
    df["MACD"] = (
        df.groupby("Ticker")["Price"].transform(lambda x: x.ewm(span=12).mean()) -
        df.groupby("Ticker")["Price"].transform(lambda x: x.ewm(span=26).mean())
    )
    df["ATR"] = df.groupby("Ticker")["Price"].transform(lambda x: x.pct_change().abs().rolling(14).mean())
    df["EMA_diff"] = df.groupby("Ticker")["Price"].transform(lambda x: x.ewm(span=5).mean() - x.ewm(span=20).mean())
    return df

def classify_volatility(std, quantiles):
    if std <= quantiles[0]:
        return 0  # Low
    elif std >= quantiles[1]:
        return 2  # High
    else:
        return 1  # Medium

def load_data():
    if not os.path.exists(DATA_FILE):
        print("❌ Data file not found.")
        return None

    df = pd.read_csv(DATA_FILE)
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    if "Sentiment" not in df.columns or "Sector" not in df.columns:
        print("⚠️ Required columns missing.")
        return None

    df = compute_indicators(df)
    df["Target_Class"] = (df.groupby("Ticker")["Price"].diff().shift(-1) > 0).astype(int)
    df["Target_Return"] = df.groupby("Ticker")["Price"].pct_change().shift(-1)
    df["Volatility"] = df.groupby("Ticker")["Price"].transform(
        lambda x: x.pct_change().rolling(5).std().shift(-1)
    )
    q_low = df["Volatility"].quantile(0.33)
    q_high = df["Volatility"].quantile(0.66)
    df["Target_Vol"] = df["Volatility"].apply(lambda x: classify_volatility(x, (q_low, q_high)))
    df = df.dropna(subset=["Target_Class", "Target_Return", "Target_Vol"])
    print(f"✅ Loaded data: {df.shape}")
    return df

def prepare_sequences(df, steps=SEQUENCE_STEPS):
    """
    Prepare LSTM-ready sequences with technical + sector features.
    """
    feature_cols = [
        "Price", "Sentiment", "MA", "STD", "RSI", "Momentum", "MACD", "ATR", "EMA_diff"
    ]
    all_X, y_class, y_reg, y_vol = [], [], [], []

    df["Sector"] = df["Sector"].astype("category")
    sector_categories = df["Sector"].cat.categories
    sector_map = {sector: idx for idx, sector in enumerate(sector_categories)}
    num_sectors = len(sector_map)

    for ticker in df["Ticker"].unique():
        sub = df[df["Ticker"] == ticker].copy().dropna(subset=feature_cols)
        if len(sub) < steps + 1:
            continue

        try:
            scaled_features = MinMaxScaler().fit_transform(sub[feature_cols])
        except:
            continue

        if np.all(scaled_features == scaled_features[0, :]):
            continue

        sector_id = sector_map.get(sub["Sector"].iloc[-1], None)
        if sector_id is None:
            continue
        sector_vec = np.zeros(num_sectors)
        sector_vec[sector_id] = 1
        sector_seq = np.repeat(sector_vec.reshape(1, -1), steps, axis=0)

        for i in range(steps, len(sub)):
            main_window = scaled_features[i - steps:i]
            full_window = np.concatenate([main_window, sector_seq], axis=1)
            all_X.append(full_window)
            y_class.append(sub["Target_Class"].iloc[i])
            y_reg.append(sub["Target_Return"].iloc[i])
            y_vol.append(sub["Target_Vol"].iloc[i])

    print(f"📦 Total sequences: {len(all_X)}")
    return (
        np.array(all_X),
        np.array(y_class),
        np.array(y_reg),
        to_categorical(y_vol, num_classes=3)
    )

def attention_block(inputs):
    """
    Self-attention mechanism.
    """
    a = Permute((2, 1))(inputs)
    a = Dense(inputs.shape[1], activation='softmax', name="attention_weights")(a)
    a = Permute((2, 1))(a)
    output = Multiply()([inputs, a])
    return output, a

def build_model(input_shape):
    """
    Builds the full model with attention + multi-head outputs.
    """
    inp = Input(shape=input_shape)

    x = LSTM(128, return_sequences=True)(inp)
    x = Dropout(0.3)(x)

    attn_out, attn_weights = attention_block(x)
    x = LSTM(64)(attn_out)
    x = Dropout(0.3)(x)

    dense_path = Dense(64, activation="relu")(x)
    dense_path = Dropout(0.2)(dense_path)

    out_class = Dense(1, activation="sigmoid", name="class_output")(dense_path)
    out_reg = Dense(1, activation="linear", name="reg_output")(dense_path)
    out_vol = Dense(3, activation="softmax", name="vol_output")(dense_path)

    model = Model(inputs=inp, outputs=[out_class, out_reg, out_vol])
    return model

def train():
    df = load_data()
    if df is None or df.empty:
        print("❌ No data for training.")
        return

    X, y_class, y_reg, y_vol = prepare_sequences(df)
    if X.shape[0] == 0:
        print("❌ No usable sequences.")
        return

    X_train, X_val, y_c_tr, y_c_val, y_r_tr, y_r_val, y_v_tr, y_v_val = train_test_split(
        X, y_class, y_reg, y_vol, test_size=0.2, random_state=SEED
    )

    model = build_model(input_shape=X.shape[1:])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss={
            "class_output": "binary_crossentropy",
            "reg_output": "mse",
            "vol_output": "categorical_crossentropy"
        },
        loss_weights={
            "class_output": 0.5,
            "reg_output": 0.3,
            "vol_output": 0.2
        },
        metrics={
            "class_output": "accuracy",
            "reg_output": "mae",
            "vol_output": "accuracy"
        }
    )

    callbacks = [
        EarlyStopping(patience=4, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=2, verbose=1, min_lr=1e-5)
    ]

    history = model.fit(
        X_train,
        {"class_output": y_c_tr, "reg_output": y_r_tr, "vol_output": y_v_tr},
        validation_data=(X_val, {
            "class_output": y_c_val,
            "reg_output": y_r_val,
            "vol_output": y_v_val
        }),
        epochs=15,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )

    os.makedirs("models", exist_ok=True)
    model.save(MODEL_FILE)
    print("✅ Model saved to:", MODEL_FILE)

    avg_loss = np.mean(history.history["loss"])
    avg_acc = np.mean(history.history.get("class_output_accuracy", [0]))
    avg_mae = np.mean(history.history.get("reg_output_mae", [0]))
    avg_vol = np.mean(history.history.get("vol_output_accuracy", [0]))

    metrics = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_points": int(len(X)),
        "input_shape": list(X.shape[1:]),
        "train_loss": float(avg_loss),
        "train_acc": float(avg_acc),
        "reg_mae": float(avg_mae),
        "vol_acc": float(avg_vol)
    }

    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"📊 Metrics logged to: {METRICS_FILE}")

if __name__ == "__main__":
    train()