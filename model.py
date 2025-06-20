import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Permute, Multiply
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from datetime import datetime
import json
import os
from tensorflow.keras.losses import MeanSquaredError

DATA_FILE = "data/sp500_data.csv"
MODEL_FILE = "models/lstm_model.keras"
METRICS_FILE = "models/metrics.json"

def rsi(series, period=14):
    """
    Calculate the Relative Strength Index (RSI) for a given series.
    """
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

def compute_indicators(df):
    """
    Compute moving average, standard deviation, and RSI for each ticker.
    """
    df['MA'] = df.groupby('Ticker')['Price'].transform(lambda x: x.rolling(5).mean())
    df['STD'] = df.groupby('Ticker')['Price'].transform(lambda x: x.rolling(5).std())
    df['RSI'] = df.groupby('Ticker')['Price'].transform(lambda x: rsi(x))
    return df

def classify_volatility(std, quantiles):
    """
    Classify volatility into three classes based on quantile thresholds.
    """
    if std <= quantiles[0]:
        return 0
    elif std >= quantiles[1]:
        return 2
    else:
        return 1

def load_data():
    if not os.path.exists(DATA_FILE):
        print("❌ Data file not found.")
        return None

    df = pd.read_csv(DATA_FILE)
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

    print("🔍 Raw data:", df.shape)

    # Ensure required columns exist
    if 'Sentiment' not in df.columns:
        print("⚠️ 'Sentiment' column missing from data.")
        return None

    df = compute_indicators(df)
    print("📉 After indicators:", df.shape)

    # Create target variables
    df['Target_Class'] = (df.groupby('Ticker')['Price'].diff().shift(-1) > 0).astype(int)
    df['Target_Return'] = df.groupby('Ticker')['Price'].pct_change().shift(-1)
    df['Volatility'] = df.groupby('Ticker')['Price'].transform(
        lambda x: x.pct_change().rolling(5).std().shift(-1)
    )
    q_low = df['Volatility'].quantile(0.33)
    q_high = df['Volatility'].quantile(0.66)
    df['Target_Vol'] = df['Volatility'].apply(lambda x: classify_volatility(x, (q_low, q_high)))

    # Drop only rows that are missing critical targets—not all indicators
    target_cols = ['Target_Class', 'Target_Return', 'Target_Vol']
    df = df.dropna(subset=target_cols)

    print("🧹 After dropna:", df.shape)
    print("📊 Top tickers by row count:")
    print(df['Ticker'].value_counts().head(10))

    return df

def prepare_sequences(df, steps=6):
    """
    Prepare sequence data for the LSTM model, including appended sector info.
    """
    base_features = ['Price', 'Sentiment', 'MA', 'STD', 'RSI']
    all_X, y_class, y_reg, y_vol = [], [], [], []

    df["Sector"] = df["Sector"].astype("category")
    sector_categories = df["Sector"].cat.categories
    sector_map = {sector: idx for idx, sector in enumerate(sector_categories)}
    n_sectors = len(sector_map)

    for ticker in df['Ticker'].unique():
        sub = df[df['Ticker'] == ticker].reset_index(drop=True)

        # Handle missing or extreme values before scaling
        sub = sub.replace([np.inf, -np.inf], np.nan)
        sub['RSI'] = sub['RSI'].fillna(0)

        safe_features = ['Price', 'Sentiment', 'MA', 'STD']  # Exclude RSI from filtering
        sub = sub.dropna(subset=safe_features)
        if len(sub) < steps + 1:
            print(f"⛔ Skipping {ticker} — only {len(sub)} usable rows after dropna")
            continue

        try:
            scaled_main = MinMaxScaler().fit_transform(sub[base_features])
        except ValueError:
            print(f"⚠️ Skipping {ticker} — scaler error")
            continue

        if np.all(scaled_main == scaled_main[0, :]):
            print(f"⚠️ Skipping {ticker} — flat input features")
            continue

        sector = sub["Sector"].iloc[-1]
        sector_idx = sector_map.get(sector)
        if sector_idx is None:
            print(f"⚠️ Skipping {ticker} — unknown sector '{sector}'")
            continue

        sector_onehot = np.zeros(n_sectors)
        sector_onehot[sector_idx] = 1
        sector_3d = np.repeat(sector_onehot.reshape(1, -1), steps, axis=0)

        seq_count = 0
        for i in range(steps, len(sub)):
            main_seq = scaled_main[i - steps:i]
            full_seq = np.concatenate([main_seq, sector_3d], axis=1)
            all_X.append(full_seq)
            y_class.append(sub['Target_Class'].iloc[i])
            y_reg.append(sub['Target_Return'].iloc[i])
            y_vol.append(sub['Target_Vol'].iloc[i])
            seq_count += 1

        if seq_count:
            print(f"✅ {ticker}: {seq_count} sequences")

    print(f"\n📦 Total sequences generated: {len(all_X)}")
    return (
        np.array(all_X),
        np.array(y_class),
        np.array(y_reg),
        to_categorical(np.array(y_vol), num_classes=3)
    )

# --- Attention Layer ---
from tensorflow.keras.layers import Lambda
import tensorflow.keras.backend as K

def attention_block(inputs):
    a = Permute((2, 1))(inputs)
    a = Dense(inputs.shape[1], activation='softmax', name="attention_weights")(a)
    a = Permute((2, 1))(a)
    out = Multiply()([inputs, a])
    return out, a  # Return both weighted output and attention weights

def build_model(input_shape):
    inp = Input(shape=input_shape)
    x = LSTM(64, return_sequences=True)(inp)
    x = Dropout(0.3)(x)
    x, attn_weights = attention_block(x)
    x = LSTM(32)(x)
    x = Dropout(0.3)(x)
    
    out_class = Dense(1, activation='sigmoid', name="class_output")(x)
    out_reg = Dense(1, activation='linear', name="reg_output")(x)
    out_vol = Dense(3, activation='softmax', name="vol_output")(x)
    
    model = Model(inputs=inp, outputs=[out_class, out_reg, out_vol])
    return model

def train():
    df = load_data()
    if df is None or df.empty:
        print("❌ No data available for training.")
        return
    
    X, y_class, y_reg, y_vol = prepare_sequences(df)
    if X.size == 0:
        print("❌ No sequence data generated. Check your data and parameters.")
        return
    
    model = build_model(X.shape[1:])
    model.compile(
        optimizer='adam',
        loss={
            'class_output': 'binary_crossentropy',
            'reg_output': MeanSquaredError(),
            'vol_output': 'categorical_crossentropy'
        },
        metrics={
            'class_output': 'accuracy',
            'reg_output': 'mae',
            'vol_output': 'accuracy'
        }
    )
    
    callbacks = [EarlyStopping(patience=3, restore_best_weights=True)]
    history = model.fit(
        X,
        {'class_output': y_class, 'reg_output': y_reg, 'vol_output': y_vol},
        batch_size=128,
        epochs=10,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    os.makedirs("models", exist_ok=True)
    model.save(MODEL_FILE)
    print("✅ Model trained and saved!")
    
    # Save training metrics
    metrics = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_rows": int(len(df)),
        "input_shape": list(X.shape[1:]),
        "train_loss": float(np.mean(history.history['loss'])),
        "train_acc": float(np.mean(history.history.get('class_output_accuracy', [0]))),
        "reg_mae": float(np.mean(history.history.get('reg_output_mae', [0]))),
        "vol_acc": float(np.mean(history.history.get('vol_output_accuracy', [0])))
    }
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"📊 Metrics saved to {METRICS_FILE}")

if __name__ == "__main__":
    train()