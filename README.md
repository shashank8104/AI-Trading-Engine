# AI Trading Intelligence System

Real-time AI-powered decision-support system for NIFTY 50 options trading (spreads).  
**Manual execution only** — this is a signal generation system, not an auto-trader.

## Architecture

```
Zerodha WebSocket → Candle Engine (5m/15m) → Feature Engine → ML Ensemble → Signal Engine → Dashboard
```

All components are async, non-blocking, and connected via `asyncio.Queue`.  
End-to-end latency target: **<1 second** after candle close.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | Python 3.10+, FastAPI (async) |
| Database | PostgreSQL + SQLAlchemy (asyncpg) |
| Data Feed | Zerodha Kite Connect WebSocket |
| ML Models | XGBoost + LSTM (PyTorch) ensemble |
| Dashboard | Streamlit + Plotly |
| Deployment | Docker Compose, AWS EC2 (Ubuntu) |

## Quick Start

### 1. Prerequisites

- Python 3.10+
- PostgreSQL 15+ (or Docker)
- Zerodha Kite Connect API credentials

### 2. Setup

```bash
git clone <repo-url>
cd ai-trading-engine
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your Kite API credentials and DB URL
```

### 3. Start PostgreSQL

```bash
docker compose up -d db
```

### 4. Generate Access Token (daily before market open)

```bash
python scripts/generate_token.py
# Follow instructions → login → paste request_token
```

### 5. Seed NFO Instruments (daily)

```bash
python scripts/seed_instruments.py
```

### 6. Fetch Historical Data (once, for training)

```bash
# From Kite API (last 60 days)
python scripts/fetch_historical.py --days 60

# Or from CSV
python scripts/fetch_historical.py --csv data/nifty_5m.csv
```

### 7. Train ML Models

```bash
python scripts/train_model.py --interval 5m --epochs 30
```

Outputs: `models/xgboost_model.pkl`, `models/lstm_model.pt`, `models/scaler.pkl`

### 8. Run the System

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 9. Launch Dashboard

```bash
streamlit run dashboard/app.py
```

### 10. Backtest (optional)

```bash
python backtesting/backtest.py --interval 5m --days 30 --verbose
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/health` | Health check |
| `GET /api/status` | Full pipeline status (WS, ML, signals) |
| `GET /api/candles/{interval}` | Recent NIFTY candles |
| `GET /api/signals/latest` | Active trade signals |
| `GET /api/signals/history` | Signal history |
| `GET /api/predictions/latest` | Latest ML ensemble prediction |

## Project Structure

```
├── app/
│   ├── main.py              # FastAPI + full pipeline wiring
│   ├── config.py             # Pydantic settings from .env
│   ├── core/
│   │   ├── data_ingest.py    # Zerodha WebSocket + dynamic ATM
│   │   ├── candle_engine.py  # Tick → OHLCV aggregation (5m/15m)
│   │   ├── feature_engine.py # 15+ technical indicators
│   │   ├── ml_engine.py      # XGBoost + LSTM ensemble
│   │   └── signal_engine.py  # Signal rules + spread recommendations
│   ├── db/
│   │   ├── database.py       # Async SQLAlchemy engine
│   │   ├── models.py         # 5 ORM tables
│   │   └── crud.py           # Async CRUD with upsert
│   ├── api/
│   │   └── routes.py         # REST endpoints
│   └── utils/
│       └── logger.py         # Structured logging
├── scripts/
│   ├── generate_token.py     # Kite access token helper
│   ├── seed_instruments.py   # NFO instrument cache
│   ├── fetch_historical.py   # Historical data loader (API + CSV)
│   └── train_model.py        # XGBoost + LSTM training pipeline
├── dashboard/
│   └── app.py                # Streamlit live dashboard
├── backtesting/
│   └── backtest.py           # Historical replay with P&L tracking
├── models/                   # Trained model artifacts (.pkl, .pt)
├── requirements.txt
├── docker-compose.yml
└── .env.example
```

## ML Pipeline

### Feature Vector (19 features)

| Group | Features |
|-------|----------|
| **Market** | RSI(14), MACD(12/26/9), Bollinger(20,2σ), ATR(14), VWAP, EMA(9,21), ADX(14), Volume ratio |
| **Options** | PCR, Max Pain, ATM OI change (CE/PE) |
| **Derived** | Body/wick ratio, gap percentage |

### Ensemble

- **XGBoost** (weight=0.6): 300 trees, max_depth=6, multi:softprob
- **LSTM** (weight=0.4): 2-layer, 64 hidden, 20-candle sequence
- **Agreement required**: Both models must agree on direction

### Signal Rules

1. Direction must be BULLISH or BEARISH (not NEUTRAL)
2. Both XGBoost and LSTM agree
3. Confidence ≥ 70% (configurable)
4. Market window: 9:30 AM – 3:00 PM IST
5. Cooldown: ≥10 minutes between signals

## Daily Workflow

1. **Pre-market (before 9:15 AM IST):**
   - Generate fresh access token
   - Seed day's instruments
   - Start the system

2. **Market hours (9:15 AM – 3:30 PM IST):**
   - Auto-connects to Zerodha WebSocket
   - Candles aggregate at 5m/15m intervals
   - Features computed → ML prediction → signal generation
   - Monitor via Streamlit dashboard

3. **Post-market:**
   - Flush remaining candle buffers
   - Data persists for future training
   - Optionally run backtest on the day's data

## License

Private — all rights reserved.
