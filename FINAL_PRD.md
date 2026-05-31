# FINAL PRD — AI Trading Intelligence System

## Objective
Build a real-time AI-powered decision-support system for NIFTY 50 options trading (spreads).
Manual execution only.

## Core Features
- Real-time data ingestion via Zerodha WebSocket
- Candle engine (5m, 15m)
- Feature engineering (market, sentiment, options)
- ML models (XGBoost + LSTM ensemble)
- Signal engine (confidence ≥ 0.7, strict agreement)
- Dashboard (Streamlit)

## Tech Stack (LOCKED)
- Python 3.10+
- FastAPI (async)
- PostgreSQL + SQLAlchemy
- XGBoost + PyTorch
- Zerodha Kite Connect (WebSocket)
- Streamlit + Plotly
- AWS EC2 (Ubuntu)
- Cron jobs
- .env config

## Constraints
- Async non-blocking system
- No Kafka/Airflow (Phase 1)
- Single-server deployment
- <1s latency after candle close

## Deliverables
- Backend pipeline
- Database
- Dashboard
- Backtesting module
- Deployment-ready system
