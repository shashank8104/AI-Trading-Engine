"""
Model training pipeline for XGBoost + LSTM.

Fetches historical candles + features from the database,
labels data by next-candle direction, trains both models,
and saves artifacts to the models/ directory.

Usage:
    python scripts/train_model.py --interval 5m
    python scripts/train_model.py --interval 15m --epochs 50

Output:
    models/xgboost_model.pkl
    models/lstm_model.pt
"""

import argparse
import asyncio
import os
import pickle
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from app.config import get_settings
from app.core.ml_engine import (
    FEATURE_COLUMNS,
    LSTM_SEQ_LEN,
    LSTMModel,
    NUM_FEATURES,
)
from app.db.crud import CandleCRUD, FeatureCRUD
from app.db.database import _get_session_factory, init_db


# ── Label Generation ──────────────────────────────────────────────────────

# Minimum price movement (% of close) to classify as bullish/bearish
DIRECTION_THRESHOLD = 0.001  # 0.1%


def label_direction(current_close: float, next_close: float) -> int:
    """
    Label the direction of the next candle.

    Returns:
        0 = BULLISH  (next_close > current_close + threshold)
        1 = BEARISH  (next_close < current_close - threshold)
        2 = NEUTRAL  (within threshold)
    """
    pct_change = (next_close - current_close) / current_close
    if pct_change > DIRECTION_THRESHOLD:
        return 0  # BULLISH
    elif pct_change < -DIRECTION_THRESHOLD:
        return 1  # BEARISH
    else:
        return 2  # NEUTRAL


# ── Data Loading ──────────────────────────────────────────────────────────


async def load_training_data(interval: str) -> pd.DataFrame:
    """
    Load candles + features from DB and create training dataset.

    Returns DataFrame with feature columns + 'label' column.
    """
    settings = get_settings()
    await init_db()

    session_factory = _get_session_factory()
    async with session_factory() as session:
        # Get all NIFTY candles
        candles = await CandleCRUD.get_recent_candles(
            session,
            settings.NIFTY_INSTRUMENT_TOKEN,
            interval,
            limit=50_000,  # All available data
        )

        # Get all features
        features = await FeatureCRUD.get_latest_features(
            session,
            settings.NIFTY_INSTRUMENT_TOKEN,
            interval,
            limit=50_000,
        )

    if len(candles) < 100:
        print(f"ERROR: Only {len(candles)} candles found. Need at least 100.")
        print("Run: python scripts/fetch_historical.py --days 60")
        sys.exit(1)

    print(f"Loaded {len(candles)} candles, {len(features)} feature rows")

    # Build candle DataFrame
    candle_df = pd.DataFrame(
        [
            {"id": c.id, "timestamp": c.timestamp, "close": c.close}
            for c in candles
        ]
    )

    # Build feature DataFrame
    if not features:
        print("No features found. Computing features from candle data...")
        # Fallback: compute features inline (simplified)
        return _compute_features_inline(candles)

    feature_rows = []
    for f in features:
        row = {"candle_id": f.candle_id}
        for col in FEATURE_COLUMNS:
            row[col] = getattr(f, col, None)
        feature_rows.append(row)

    feature_df = pd.DataFrame(feature_rows)

    # Merge on candle_id
    merged = candle_df.merge(feature_df, left_on="id", right_on="candle_id", how="inner")
    merged.sort_values("timestamp", inplace=True)
    merged.reset_index(drop=True, inplace=True)

    # Generate labels (next-candle direction)
    labels = []
    for i in range(len(merged) - 1):
        labels.append(label_direction(merged.loc[i, "close"], merged.loc[i + 1, "close"]))
    labels.append(2)  # Last row has no "next" candle

    merged["label"] = labels

    # Drop the last row (no valid label)
    merged = merged.iloc[:-1]

    # Fill NaN features with 0
    for col in FEATURE_COLUMNS:
        merged[col] = merged[col].fillna(0).astype(float)

    return merged


def _compute_features_inline(candles) -> pd.DataFrame:
    """
    Fallback: compute features directly from candles when features table is empty.
    Uses a simplified set of indicators.
    """
    from ta.momentum import RSIIndicator
    from ta.trend import MACD, EMAIndicator, ADXIndicator
    from ta.volatility import BollingerBands, AverageTrueRange

    df = pd.DataFrame(
        [
            {
                "timestamp": c.timestamp,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
            }
            for c in candles
        ]
    )
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # Compute indicators
    rsi = RSIIndicator(close=close, window=14)
    macd_ind = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    bb = BollingerBands(close=close, window=20, window_dev=2)
    atr = AverageTrueRange(high=high, low=low, close=close, window=14)
    ema9 = EMAIndicator(close=close, window=9)
    ema21 = EMAIndicator(close=close, window=21)
    adx = ADXIndicator(high=high, low=low, close=close, window=14)

    df["rsi_14"] = rsi.rsi()
    df["macd"] = macd_ind.macd()
    df["macd_signal"] = macd_ind.macd_signal()
    df["macd_histogram"] = macd_ind.macd_diff()
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_middle"] = bb.bollinger_mavg()
    df["bb_lower"] = bb.bollinger_lband()
    df["atr_14"] = atr.average_true_range()

    typical = (high + low + close) / 3
    cum_vol = volume.cumsum().replace(0, np.nan)
    df["vwap"] = (typical * volume).cumsum() / cum_vol

    df["ema_9"] = ema9.ema_indicator()
    df["ema_21"] = ema21.ema_indicator()

    vol_sma = volume.rolling(20).mean().replace(0, np.nan)
    df["volume_sma_ratio"] = volume / vol_sma

    df["adx"] = adx.adx()

    # Options features (set to 0 — no OI data during offline training)
    df["pcr"] = 0.0
    df["max_pain"] = 0.0
    df["atm_oi_change_ce"] = 0.0
    df["atm_oi_change_pe"] = 0.0

    # Derived features
    body = (df["close"] - df["open"]).abs()
    upper_wick = df["high"] - df[["close", "open"]].max(axis=1)
    lower_wick = df[["close", "open"]].min(axis=1) - df["low"]
    total_wick = upper_wick + lower_wick
    df["body_wick_ratio"] = np.where(total_wick > 0, body / total_wick, 10.0)
    df["gap_pct"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1) * 100

    # Labels
    labels = []
    for i in range(len(df) - 1):
        labels.append(label_direction(df.loc[i, "close"], df.loc[i + 1, "close"]))
    labels.append(2)
    df["label"] = labels

    # Drop warmup rows and last row
    df = df.iloc[30:-1].reset_index(drop=True)

    # Fill NaN
    for col in FEATURE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(float)
        else:
            df[col] = 0.0

    return df


# ── XGBoost Training ─────────────────────────────────────────────────────


def train_xgboost(X_train, y_train, X_val, y_val) -> XGBClassifier:
    """Train XGBoost classifier."""
    print("\n" + "=" * 60)
    print("  Training XGBoost")
    print("=" * 60)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    # Evaluate
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"\n  Validation Accuracy: {acc:.4f}")
    print(classification_report(y_val, y_pred, target_names=["BULLISH", "BEARISH", "NEUTRAL"]))

    return model


# ── LSTM Training ─────────────────────────────────────────────────────────


def create_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    """Create sequences for LSTM input."""
    X_seq, y_seq = [], []
    for i in range(seq_len, len(X)):
        X_seq.append(X[i - seq_len : i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)


def train_lstm(
    X_train_seq: np.ndarray,
    y_train_seq: np.ndarray,
    X_val_seq: np.ndarray,
    y_val_seq: np.ndarray,
    epochs: int = 30,
) -> LSTMModel:
    """Train LSTM model."""
    print("\n" + "=" * 60)
    print("  Training LSTM")
    print("=" * 60)

    device = torch.device("cpu")
    model = LSTMModel().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Convert to tensors
    X_train_t = torch.tensor(X_train_seq, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_seq, dtype=torch.long)
    X_val_t = torch.tensor(X_val_seq, dtype=torch.float32)
    y_val_t = torch.tensor(y_val_seq, dtype=torch.long)

    # Training loop
    batch_size = 64
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        # Mini-batch training
        indices = torch.randperm(len(X_train_t))
        for start in range(0, len(X_train_t), batch_size):
            idx = indices[start : start + batch_size]
            X_batch = X_train_t[idx]
            y_batch = y_train_t[idx]

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_preds = val_outputs.argmax(dim=1)
            val_acc = (val_preds == y_val_t).float().mean().item()

        avg_loss = total_loss / max(n_batches, 1)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch + 1:3d}/{epochs} — "
                f"Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc

    print(f"\n  Best Validation Accuracy: {best_val_acc:.4f}")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_t)
        val_preds = val_outputs.argmax(dim=1).numpy()
    print(classification_report(y_val_seq, val_preds, target_names=["BULLISH", "BEARISH", "NEUTRAL"]))

    return model


# ── Main ──────────────────────────────────────────────────────────────────


async def run_training(interval: str, epochs: int) -> None:
    """Full training pipeline."""
    print(f"\n{'=' * 60}")
    print(f"  AI Trading Model Training Pipeline")
    print(f"  Interval: {interval} | LSTM Epochs: {epochs}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}")

    # 1. Load data
    df = await load_training_data(interval)
    print(f"\nDataset: {len(df)} samples")
    print(f"Label distribution:")
    for label, name in enumerate(["BULLISH", "BEARISH", "NEUTRAL"]):
        count = (df["label"] == label).sum()
        print(f"  {name}: {count} ({count / len(df) * 100:.1f}%)")

    # 2. Prepare features and labels
    X = df[FEATURE_COLUMNS].values.astype(np.float32)
    y = df["label"].values.astype(np.int64)

    # Replace NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # 3. Time-based train/val split (80/20, no shuffle)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print(f"\nTrain: {len(X_train)} | Validation: {len(X_val)}")

    # 4. Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # 5. Train XGBoost (on unscaled data — tree models don't need scaling)
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val)

    # 6. Create LSTM sequences (on scaled data)
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, LSTM_SEQ_LEN)
    X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val, LSTM_SEQ_LEN)

    if len(X_train_seq) < 10:
        print(f"\nWARNING: Only {len(X_train_seq)} sequences for LSTM. Need more data.")
    else:
        print(f"\nLSTM sequences — Train: {len(X_train_seq)} | Val: {len(X_val_seq)}")

    # 7. Train LSTM
    if len(X_train_seq) >= 10:
        lstm_model = train_lstm(X_train_seq, y_train_seq, X_val_seq, y_val_seq, epochs)
    else:
        print("Skipping LSTM training — insufficient data")
        lstm_model = LSTMModel()

    # 8. Save models
    settings = get_settings()
    os.makedirs(os.path.dirname(settings.XGBOOST_MODEL_PATH) or "models", exist_ok=True)

    with open(settings.XGBOOST_MODEL_PATH, "wb") as f:
        pickle.dump(xgb_model, f)
    print(f"\nDone: XGBoost saved: {settings.XGBOOST_MODEL_PATH}")

    torch.save(lstm_model.state_dict(), settings.LSTM_MODEL_PATH)
    print(f"Done: LSTM saved: {settings.LSTM_MODEL_PATH}")

    # Save scaler for inference
    scaler_path = os.path.join(os.path.dirname(settings.XGBOOST_MODEL_PATH), "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Done: Scaler saved: {scaler_path}")

    print(f"\n{'=' * 60}")
    print(f"  Training complete!")
    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(description="Train ML models")
    parser.add_argument(
        "--interval",
        type=str,
        default="5m",
        choices=["5m", "15m"],
        help="Candle interval to train on",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="LSTM training epochs",
    )
    args = parser.parse_args()
    asyncio.run(run_training(args.interval, args.epochs))


if __name__ == "__main__":
    main()
