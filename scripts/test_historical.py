import functools, datetime
from app.config import get_settings
from scripts.fetch_historical import _get_smart_client

s = get_settings()
smart = _get_smart_client(s)

p = {
    'exchange': 'NSE',
    'symboltoken': '99926000',
    'interval': 'FIVE_MINUTE',
    'fromdate': '2026-05-25 09:15',
    'todate': '2026-05-30 15:30'
}

r = smart.getCandleData(p)
if r and r.get('data'):
    print(f"Got {len(r['data'])} candles")
    print(f"Sample: {r['data'][:2]}")
else:
    print(f"No data. Response: {r}")
