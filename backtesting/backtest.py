"""
Backtesting module for the AI Trading Intelligence System.

Replays historical candles through the full pipeline:
    candles → features → ML prediction → signal generation → P&L tracking

Usage:
    python backtesting/backtest.py --interval 5m --days 30
    python backtesting/backtest.py --interval 15m --days 60 --verbose

Output:
    - Win rate, avg return, max drawdown, Sharpe ratio
    - Per-signal detail log
"""

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

import numpy as np
import pandas as pd

from app.config import get_settings
from app.core.feature_engine import FeatureEngine
from app.core.ml_engine import MLEngine, FEATURE_COLUMNS
from app.core.signal_engine import SignalEngine, TradeSignal
from app.db.crud import CandleCRUD
from app.db.database import _get_session_factory, init_db
from app.db.models import Candle


# ── Trade Result ──────────────────────────────────────────────────────────


@dataclass
class TradeResult:
    """Record of a backtested trade."""

    entry_time: datetime
    exit_time: Optional[datetime]
    direction: str
    spread_type: str
    entry_price: float
    exit_price: float
    stop_loss: float
    target: float
    confidence: float
    pnl_pct: float
    outcome: str  # WIN | LOSS | OPEN


@dataclass
class BacktestResult:
    """Aggregate backtest statistics."""

    total_signals: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    avg_confidence: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    win_rate: float = 0.0
    trades: List[TradeResult] = field(default_factory=list)


# ── Backtester ────────────────────────────────────────────────────────────


class Backtester:
    """
    Replays historical data through the trading pipeline.

    Uses the actual ML models and signal engine rules
    to simulate what would have happened in production.
    """

    def __init__(self, interval: str = "5m", verbose: bool = False):
        self.interval = interval
        self.verbose = verbose
        self.settings = get_settings()
        self.ml_engine = MLEngine()
        self.signal_engine = SignalEngine()
        self.feature_engine = FeatureEngine()

    async def run(self, days: int) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            days: number of days of history to backtest

        Returns:
            BacktestResult with all statistics and trade details
        """
        print(f"\n{'=' * 60}")
        print(f"  BACKTESTING - {self.interval} candles, last {days} days")
        print(f"{'=' * 60}\n")

        # Load models
        models_loaded = self.ml_engine.load_models()
        if not models_loaded:
            print("WARNING: No trained models found. Using random predictions.")
            print("Run: python scripts/train_model.py --interval", self.interval)

        # Load historical candles
        await init_db()
        candles = await self._load_candles(days)

        if len(candles) < 50:
            print(f"ERROR: Only {len(candles)} candles. Need at least 50.")
            return BacktestResult()

        print(f"Loaded {len(candles)} candles")
        print(f"Period: {candles[0].timestamp} -> {candles[-1].timestamp}")
        print()

        # Simulate the pipeline
        result = await self._simulate(candles)

        # Print results
        self._print_results(result)

        return result

    async def _load_candles(self, days: int) -> List[Candle]:
        """Load candles from database."""
        session_factory = _get_session_factory()
        async with session_factory() as session:
            candles = await CandleCRUD.get_recent_candles(
                session,
                self.settings.NIFTY_INSTRUMENT_TOKEN,
                self.interval,
                limit=days * 75,  # ~75 candles per day for 5m
            )
        return candles

    async def _simulate(self, candles: List[Candle]) -> BacktestResult:
        """
        Simulate the trading pipeline on historical candles.

        For each candle (beyond warmup):
            1. Compute features from candle history
            2. Run ML prediction
            3. Evaluate signal rules
            4. Track open trades for P&L
        """
        result = BacktestResult()
        open_trade: Optional[TradeSignal] = None
        open_trade_entry_idx: int = 0
        equity_curve: List[float] = [0.0]
        feature_history: List[np.ndarray] = []

        # We need at least 50 candles for feature computation
        warmup = 50

        # Process candles after warmup
        for i in range(warmup, len(candles)):
            candle = candles[i]
            history = candles[max(0, i - 50) : i + 1]

            # Check if open trade hits SL or target
            if open_trade is not None:
                trade_result = self._check_trade_exit(
                    open_trade, candle, open_trade_entry_idx, i
                )
                if trade_result is not None:
                    result.trades.append(trade_result)
                    equity_curve.append(equity_curve[-1] + trade_result.pnl_pct)

                    if trade_result.outcome == "WIN":
                        result.winning_trades += 1
                    else:
                        result.losing_trades += 1

                    open_trade = None

                    if self.verbose:
                        print(
                            f"  {trade_result.outcome}: "
                            f"{trade_result.direction} "
                            f"P&L: {trade_result.pnl_pct:+.2f}% "
                            f"at {candle.timestamp}"
                        )

            # Skip if already in a trade
            if open_trade is not None:
                continue

            # Build candle dict for signal engine
            candle_dict = {
                "instrument_token": candle.instrument_token,
                "interval": self.interval,
                "timestamp": candle.timestamp,
                "open": candle.open,
                "high": candle.high,
                "low": candle.low,
                "close": candle.close,
                "volume": candle.volume,
                "oi": candle.oi,
            }

            # Compute features using history
            df = self.feature_engine._candles_to_df(history)
            if len(df) < 20:
                continue

            features = self.feature_engine._compute_market_features(df)
            features.update(self.feature_engine._compute_options_features(candle_dict))
            features.update(self.feature_engine._compute_derived_features(df))
            atr = features.get("atr_14")

            # ML prediction (if models loaded)
            if self.ml_engine.is_loaded:
                feature_row = np.array(
                    [float(features.get(col) or 0.0) for col in FEATURE_COLUMNS],
                    dtype=np.float32,
                )
                feature_row = np.nan_to_num(
                    feature_row, nan=0.0, posinf=0.0, neginf=0.0
                )
                feature_history.append(feature_row)
                history_matrix = np.array(feature_history, dtype=np.float32)
                prediction = self._predict_from_history(history_matrix)
            else:
                # Fallback: simple momentum signal
                prediction = self._simple_momentum_signal(history)

            if prediction is None:
                continue

            # Signal engine evaluation
            signal = await self.signal_engine.process_prediction(
                prediction, candle_dict, atr=atr
            )

            if signal is not None:
                open_trade = signal
                open_trade_entry_idx = i
                result.total_signals += 1

                if self.verbose:
                    print(
                        f"  SIGNAL: {signal.direction} "
                        f"conf={signal.confidence:.2f} "
                        f"entry={signal.entry_price:,.2f} "
                        f"at {candle.timestamp}"
                    )

        # Close any remaining open trade at last candle
        if open_trade is not None:
            last = candles[-1]
            pnl_pct = self._calc_pnl(open_trade.direction, open_trade.entry_price, last.close)
            result.trades.append(
                TradeResult(
                    entry_time=open_trade.timestamp,
                    exit_time=last.timestamp,
                    direction=open_trade.direction,
                    spread_type=open_trade.spread_type,
                    entry_price=open_trade.entry_price,
                    exit_price=last.close,
                    stop_loss=open_trade.stop_loss,
                    target=open_trade.target,
                    confidence=open_trade.confidence,
                    pnl_pct=pnl_pct,
                    outcome="WIN" if pnl_pct > 0 else "LOSS",
                )
            )

        # Compute aggregate stats
        self._compute_stats(result, equity_curve)

        return result

    def _predict_from_history(self, feature_history: np.ndarray):
        """
        Generate ensemble prediction from in-memory feature history.

        This avoids dependence on the DB `features` table during backtests.
        """
        if feature_history.size == 0:
            return None
        xgb_probs = self.ml_engine._predict_xgboost(feature_history[-1:])
        lstm_probs = self.ml_engine._predict_lstm(feature_history)
        return self.ml_engine._ensemble(xgb_probs, lstm_probs)

    def _check_trade_exit(
        self,
        trade: TradeSignal,
        candle: Candle,
        entry_idx: int,
        current_idx: int,
    ) -> Optional[TradeResult]:
        """Check if current candle triggers SL or target."""
        if trade.direction == "BULLISH":
            # Check stop loss (price went below SL)
            if candle.low <= trade.stop_loss:
                pnl_pct = self._calc_pnl("BULLISH", trade.entry_price, trade.stop_loss)
                return TradeResult(
                    entry_time=trade.timestamp,
                    exit_time=candle.timestamp,
                    direction="BULLISH",
                    spread_type=trade.spread_type,
                    entry_price=trade.entry_price,
                    exit_price=trade.stop_loss,
                    stop_loss=trade.stop_loss,
                    target=trade.target,
                    confidence=trade.confidence,
                    pnl_pct=pnl_pct,
                    outcome="LOSS",
                )
            # Check target
            if candle.high >= trade.target:
                pnl_pct = self._calc_pnl("BULLISH", trade.entry_price, trade.target)
                return TradeResult(
                    entry_time=trade.timestamp,
                    exit_time=candle.timestamp,
                    direction="BULLISH",
                    spread_type=trade.spread_type,
                    entry_price=trade.entry_price,
                    exit_price=trade.target,
                    stop_loss=trade.stop_loss,
                    target=trade.target,
                    confidence=trade.confidence,
                    pnl_pct=pnl_pct,
                    outcome="WIN",
                )
        else:  # BEARISH
            if candle.high >= trade.stop_loss:
                pnl_pct = self._calc_pnl("BEARISH", trade.entry_price, trade.stop_loss)
                return TradeResult(
                    entry_time=trade.timestamp,
                    exit_time=candle.timestamp,
                    direction="BEARISH",
                    spread_type=trade.spread_type,
                    entry_price=trade.entry_price,
                    exit_price=trade.stop_loss,
                    stop_loss=trade.stop_loss,
                    target=trade.target,
                    confidence=trade.confidence,
                    pnl_pct=pnl_pct,
                    outcome="LOSS",
                )
            if candle.low <= trade.target:
                pnl_pct = self._calc_pnl("BEARISH", trade.entry_price, trade.target)
                return TradeResult(
                    entry_time=trade.timestamp,
                    exit_time=candle.timestamp,
                    direction="BEARISH",
                    spread_type=trade.spread_type,
                    entry_price=trade.entry_price,
                    exit_price=trade.target,
                    stop_loss=trade.stop_loss,
                    target=trade.target,
                    confidence=trade.confidence,
                    pnl_pct=pnl_pct,
                    outcome="WIN",
                )

        # Max hold: 15 candles (75 min for 5m, 225 min for 15m)
        if current_idx - entry_idx >= 15:
            exit_price = candle.close
            pnl_pct = self._calc_pnl(trade.direction, trade.entry_price, exit_price)
            return TradeResult(
                entry_time=trade.timestamp,
                exit_time=candle.timestamp,
                direction=trade.direction,
                spread_type=trade.spread_type,
                entry_price=trade.entry_price,
                exit_price=exit_price,
                stop_loss=trade.stop_loss,
                target=trade.target,
                confidence=trade.confidence,
                pnl_pct=pnl_pct,
                outcome="WIN" if pnl_pct > 0 else "LOSS",
            )

        return None

    @staticmethod
    def _calc_pnl(direction: str, entry: float, exit_price: float) -> float:
        """Calculate P&L percentage."""
        if entry == 0:
            return 0.0
        if direction == "BULLISH":
            return round((exit_price - entry) / entry * 100, 4)
        else:
            return round((entry - exit_price) / entry * 100, 4)

    def _simple_momentum_signal(self, candles):
        """Fallback signal when no ML models are available."""
        from app.core.ml_engine import PredictionResult

        if len(candles) < 10:
            return None

        recent = [c.close for c in candles[-10:]]
        sma = sum(recent) / len(recent)
        current = recent[-1]

        if current > sma * 1.002:
            return PredictionResult(
                xgboost_probs={"BULLISH": 0.6, "BEARISH": 0.2, "NEUTRAL": 0.2},
                lstm_probs={"BULLISH": 0.55, "BEARISH": 0.25, "NEUTRAL": 0.2},
                ensemble_probs={"BULLISH": 0.58, "BEARISH": 0.22, "NEUTRAL": 0.2},
                ensemble_direction="BULLISH",
                ensemble_confidence=0.58,
                models_agree=True,
            )
        elif current < sma * 0.998:
            return PredictionResult(
                xgboost_probs={"BULLISH": 0.2, "BEARISH": 0.6, "NEUTRAL": 0.2},
                lstm_probs={"BULLISH": 0.25, "BEARISH": 0.55, "NEUTRAL": 0.2},
                ensemble_probs={"BULLISH": 0.22, "BEARISH": 0.58, "NEUTRAL": 0.2},
                ensemble_direction="BEARISH",
                ensemble_confidence=0.58,
                models_agree=True,
            )
        return None

    @staticmethod
    def _compute_stats(result: BacktestResult, equity_curve: List[float]) -> None:
        """Compute aggregate statistics."""
        if not result.trades:
            return

        result.total_pnl_pct = sum(t.pnl_pct for t in result.trades)
        result.avg_confidence = np.mean([t.confidence for t in result.trades])

        wins = [t.pnl_pct for t in result.trades if t.outcome == "WIN"]
        losses = [t.pnl_pct for t in result.trades if t.outcome == "LOSS"]

        result.avg_win_pct = np.mean(wins) if wins else 0.0
        result.avg_loss_pct = np.mean(losses) if losses else 0.0

        total_closed = result.winning_trades + result.losing_trades
        result.win_rate = (
            result.winning_trades / total_closed * 100 if total_closed > 0 else 0.0
        )

        # Max drawdown
        peak = 0.0
        max_dd = 0.0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = peak - eq
            if dd > max_dd:
                max_dd = dd
        result.max_drawdown_pct = max_dd

        # Sharpe ratio (annualized, assuming 252 trading days)
        if len(result.trades) >= 2:
            returns = [t.pnl_pct for t in result.trades]
            mean_ret = np.mean(returns)
            std_ret = np.std(returns)
            if std_ret > 0:
                result.sharpe_ratio = round(
                    (mean_ret / std_ret) * np.sqrt(252), 2
                )

    def _print_results(self, result: BacktestResult) -> None:
        """Print backtest results."""
        print(f"\n{'=' * 60}")
        print(f"  BACKTEST RESULTS")
        print(f"{'=' * 60}")
        print(f"  Total Signals     : {result.total_signals}")
        print(f"  Winning Trades    : {result.winning_trades}")
        print(f"  Losing Trades     : {result.losing_trades}")
        print(f"  Win Rate          : {result.win_rate:.1f}%")
        print("  -------------------------------------")
        print(f"  Total P&L         : {result.total_pnl_pct:+.2f}%")
        print(f"  Avg Win           : {result.avg_win_pct:+.2f}%")
        print(f"  Avg Loss          : {result.avg_loss_pct:+.2f}%")
        print(f"  Max Drawdown      : {result.max_drawdown_pct:.2f}%")
        print(f"  Sharpe Ratio      : {result.sharpe_ratio:.2f}")
        print(f"  Avg Confidence    : {result.avg_confidence:.2f}")
        print(f"{'=' * 60}\n")

        if result.trades and self.verbose:
            print("  Trade Log:")
            print(f"  {'Time':<20} {'Dir':<8} {'Entry':>10} {'Exit':>10} {'P&L':>8} {'Result':<6}")
            print(f"  {'-' * 62}")
            for t in result.trades:
                print(
                    f"  {str(t.entry_time):<20} {t.direction:<8} "
                    f"{t.entry_price:>10,.2f} {t.exit_price:>10,.2f} "
                    f"{t.pnl_pct:>+7.2f}% {t.outcome:<6}"
                )


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Backtest trading strategy")
    parser.add_argument(
        "--interval", type=str, default="5m", choices=["5m", "15m"]
    )
    parser.add_argument(
        "--days", type=int, default=30, help="Days of history to backtest"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print per-trade details"
    )
    args = parser.parse_args()

    bt = Backtester(interval=args.interval, verbose=args.verbose)
    asyncio.run(bt.run(args.days))


if __name__ == "__main__":
    main()
