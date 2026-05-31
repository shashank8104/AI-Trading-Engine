"""
ML Ensemble: XGBoost + LSTM for trade direction prediction.

Architecture:
    XGBoost  — tabular model on latest feature vector (fast, <10ms)
    LSTM     — sequential model on last N feature vectors (pattern capture)
    Ensemble — weighted average; both must agree on direction

Output: 3-class probabilities (bullish, bearish, neutral) + confidence.
"""

import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from app.config import get_settings
from app.db.crud import FeatureCRUD
from app.db.database import _get_session_factory
from app.db.models import Feature
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ── Feature definition (shared with training) ────────────────────────────

FEATURE_COLUMNS = [
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_histogram",
    "bb_upper",
    "bb_middle",
    "bb_lower",
    "atr_14",
    "vwap",
    "ema_9",
    "ema_21",
    "volume_sma_ratio",
    "adx",
    "pcr",
    "max_pain",
    "atm_oi_change_ce",
    "atm_oi_change_pe",
    "body_wick_ratio",
    "gap_pct",
]

NUM_FEATURES = len(FEATURE_COLUMNS)
LSTM_SEQ_LEN = 20  # Number of past candles for LSTM input
CLASS_NAMES = ["BULLISH", "BEARISH", "NEUTRAL"]


# ── LSTM Model Definition ────────────────────────────────────────────────


class LSTMModel(nn.Module):
    """
    2-layer LSTM for sequential pattern recognition.

    Input:  (batch, seq_len, num_features)
    Output: (batch, 3) — softmax probabilities
    """

    def __init__(
        self,
        input_size: int = NUM_FEATURES,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Take the last timestep output
        last_hidden = lstm_out[:, -1, :]
        logits = self.fc(last_hidden)
        return logits


# ── Prediction Result ────────────────────────────────────────────────────


@dataclass
class PredictionResult:
    """Container for ensemble prediction output."""

    xgboost_probs: Dict[str, float]  # {BULLISH: 0.6, BEARISH: 0.3, NEUTRAL: 0.1}
    lstm_probs: Dict[str, float]
    ensemble_probs: Dict[str, float]
    ensemble_direction: str  # BULLISH | BEARISH | NEUTRAL
    ensemble_confidence: float  # max ensemble probability
    models_agree: bool  # True if XGBoost and LSTM agree on direction


# ── ML Engine ────────────────────────────────────────────────────────────


class MLEngine:
    """
    Loads and runs the XGBoost + LSTM ensemble.

    Models are loaded from disk at startup. If model files don't exist,
    the engine operates in passthrough mode (no predictions).
    """

    def __init__(self):
        self.settings = get_settings()
        self._xgb_model: Optional[xgb.XGBClassifier] = None
        self._lstm_model: Optional[LSTMModel] = None
        self._scaler: Optional[StandardScaler] = None
        self._device = torch.device("cpu")  # CPU inference for latency
        self._loaded = False

    def load_models(self) -> bool:
        """
        Load pre-trained models from disk.

        Returns True if at least one model loaded successfully.
        """
        xgb_loaded = self._load_xgboost()
        lstm_loaded = self._load_lstm()
        self._load_scaler()
        self._loaded = xgb_loaded or lstm_loaded

        if self._loaded:
            logger.info(
                "models_loaded", xgboost=xgb_loaded, lstm=lstm_loaded
            )
        else:
            logger.warning(
                "no_models_found",
                hint="Run scripts/train_model.py to train models",
            )

        return self._loaded

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    async def predict(
        self,
        instrument_token: int,
        interval: str,
    ) -> Optional[PredictionResult]:
        """
        Run ensemble prediction using latest features from DB.

        Returns PredictionResult or None if prediction can't be made.
        """
        if not self._loaded:
            return None

        # Fetch recent features for both models
        session_factory = _get_session_factory()
        async with session_factory() as session:
            features = await FeatureCRUD.get_latest_features(
                session, instrument_token, interval, limit=LSTM_SEQ_LEN
            )

        if not features:
            logger.warning("no_features_for_prediction")
            return None

        # Convert features to numpy
        feature_matrix = self._features_to_matrix(features)
        if feature_matrix is None:
            return None

        # XGBoost prediction (latest row only)
        xgb_probs = self._predict_xgboost(feature_matrix[-1:])

        # LSTM prediction (full sequence)
        lstm_probs = self._predict_lstm(feature_matrix)

        # Ensemble
        return self._ensemble(xgb_probs, lstm_probs)

    # ── Model Loading ─────────────────────────────────────────────

    def _load_xgboost(self) -> bool:
        """Load XGBoost model from pickle."""
        path = self.settings.XGBOOST_MODEL_PATH
        if not os.path.exists(path):
            logger.info("xgboost_model_not_found", path=path)
            return False
        try:
            with open(path, "rb") as f:
                self._xgb_model = pickle.load(f)
            logger.info("xgboost_model_loaded", path=path)
            return True
        except Exception as e:
            logger.error("xgboost_load_error", error=str(e))
            return False

    def _load_lstm(self) -> bool:
        """Load LSTM model from PyTorch checkpoint."""
        path = self.settings.LSTM_MODEL_PATH
        if not os.path.exists(path):
            logger.info("lstm_model_not_found", path=path)
            return False
        try:
            self._lstm_model = LSTMModel()
            state_dict = torch.load(path, map_location=self._device, weights_only=True)
            self._lstm_model.load_state_dict(state_dict)
            self._lstm_model.eval()
            logger.info("lstm_model_loaded", path=path)
            return True
        except Exception as e:
            logger.error("lstm_load_error", error=str(e))
            self._lstm_model = None
            return False

    def _load_scaler(self) -> bool:
        """
        Load the feature scaler used during training (for LSTM only).

        Training scales LSTM inputs, while XGBoost uses raw features.
        """
        try:
            model_dir = os.path.dirname(self.settings.XGBOOST_MODEL_PATH) or "models"
            scaler_path = os.path.join(model_dir, "scaler.pkl")
            if not os.path.exists(scaler_path):
                self._scaler = None
                logger.info("scaler_not_found", path=scaler_path)
                return False
            with open(scaler_path, "rb") as f:
                self._scaler = pickle.load(f)
            logger.info("scaler_loaded", path=scaler_path)
            return True
        except Exception as e:
            self._scaler = None
            logger.error("scaler_load_error", error=str(e))
            return False

    # ── Feature Extraction ────────────────────────────────────────

    def _features_to_matrix(
        self, features: List[Feature]
    ) -> Optional[np.ndarray]:
        """
        Convert Feature ORM objects to a numpy matrix.

        Shape: (num_candles, NUM_FEATURES)
        Missing values filled with 0.
        """
        if not features:
            return None

        rows = []
        for f in features:
            row = []
            for col in FEATURE_COLUMNS:
                val = getattr(f, col, None)
                row.append(float(val) if val is not None else 0.0)
            rows.append(row)

        matrix = np.array(rows, dtype=np.float32)

        # Replace any NaN/Inf with 0
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

        return matrix

    # ── Individual Model Predictions ──────────────────────────────

    def _predict_xgboost(self, features: np.ndarray) -> Dict[str, float]:
        """
        XGBoost prediction on a single feature vector.

        Returns dict of class probabilities.
        """
        if self._xgb_model is None:
            return {c: 1.0 / len(CLASS_NAMES) for c in CLASS_NAMES}

        try:
            probs = self._xgb_model.predict_proba(features)[0]
            return {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
        except Exception as e:
            logger.error("xgboost_predict_error", error=str(e))
            return {c: 1.0 / len(CLASS_NAMES) for c in CLASS_NAMES}

    def _predict_lstm(self, features: np.ndarray) -> Dict[str, float]:
        """
        LSTM prediction on a sequence of feature vectors.

        Pads or truncates to LSTM_SEQ_LEN.
        Returns dict of class probabilities.
        """
        if self._lstm_model is None:
            return {c: 1.0 / len(CLASS_NAMES) for c in CLASS_NAMES}

        try:
            # Take the most recent rows
            if len(features) > LSTM_SEQ_LEN:
                seq = features[-LSTM_SEQ_LEN:]
            else:
                seq = features

            # Apply training-time scaling for LSTM if available
            if self._scaler is not None:
                seq = self._scaler.transform(seq).astype(np.float32, copy=False)

            # Pad to fixed length (pad AFTER scaling to keep pad as zeros)
            if len(seq) < LSTM_SEQ_LEN:
                pad = np.zeros((LSTM_SEQ_LEN - len(seq), NUM_FEATURES), dtype=np.float32)
                seq = np.vstack([pad, seq])

            # Convert to tensor: (1, seq_len, features)
            tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                logits = self._lstm_model(tensor)
                probs = torch.softmax(logits, dim=1).squeeze().numpy()

            return {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
        except Exception as e:
            logger.error("lstm_predict_error", error=str(e))
            return {c: 1.0 / len(CLASS_NAMES) for c in CLASS_NAMES}

    # ── Ensemble ──────────────────────────────────────────────────

    def _ensemble(
        self,
        xgb_probs: Dict[str, float],
        lstm_probs: Dict[str, float],
    ) -> PredictionResult:
        """
        Combine XGBoost and LSTM predictions with weighted averaging.

        Weights: XGBoost=0.6, LSTM=0.4 (tunable in config).
        """
        w_xgb = self.settings.XGBOOST_WEIGHT
        w_lstm = self.settings.LSTM_WEIGHT

        # Weighted average
        ensemble_probs = {}
        for cls in CLASS_NAMES:
            ensemble_probs[cls] = round(
                w_xgb * xgb_probs[cls] + w_lstm * lstm_probs[cls], 6
            )

        # Determine directions
        xgb_direction = max(xgb_probs, key=xgb_probs.get)
        lstm_direction = max(lstm_probs, key=lstm_probs.get)
        ensemble_direction = max(ensemble_probs, key=ensemble_probs.get)
        ensemble_confidence = ensemble_probs[ensemble_direction]
        models_agree = xgb_direction == lstm_direction

        logger.info(
            "ensemble_prediction",
            xgb=xgb_direction,
            lstm=lstm_direction,
            ensemble=ensemble_direction,
            confidence=round(ensemble_confidence, 4),
            agree=models_agree,
        )

        return PredictionResult(
            xgboost_probs=xgb_probs,
            lstm_probs=lstm_probs,
            ensemble_probs=ensemble_probs,
            ensemble_direction=ensemble_direction,
            ensemble_confidence=ensemble_confidence,
            models_agree=models_agree,
        )
