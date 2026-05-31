"""
Angel One SmartAPI — Automated session generator.

Runs fully headless using TOTP (no browser, no manual step).
Reads credentials from .env, generates a fresh JWT session,
and prints the tokens for inspection.

Usage:
    python scripts/generate_token.py

Credentials required in .env:
    ANGEL_API_KEY
    ANGEL_CLIENT_ID
    ANGEL_PASSWORD
    ANGEL_TOTP_SECRET   <- base32 text key from smartapi.angelbroking.com/enable-totp
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

import pyotp

from SmartApi import SmartConnect


def generate_session() -> dict:
    """
    Generate a fresh Angel One session using TOTP.

    Returns:
        dict with keys: auth_token, feed_token, refresh_token, client_code
    """
    api_key = os.environ.get("ANGEL_API_KEY", "").strip()
    client_id = os.environ.get("ANGEL_CLIENT_ID", "").strip()
    mpin = os.environ.get("ANGEL_MPIN", "").strip()
    totp_secret = os.environ.get("ANGEL_TOTP_SECRET", "").strip()

    if not all([api_key, client_id, mpin, totp_secret]):
        print("ERROR: Missing Angel One credentials in .env")
        print("Required: ANGEL_API_KEY, ANGEL_CLIENT_ID, ANGEL_MPIN, ANGEL_TOTP_SECRET")
        sys.exit(1)

    # Generate 6-digit TOTP code from the base32 secret
    totp_code = pyotp.TOTP(totp_secret).now()

    smart = SmartConnect(api_key=api_key)

    try:
        data = smart.generateSession(client_id, mpin, totp_code)
    except Exception as e:
        print(f"ERROR: Session generation failed — {e}")
        sys.exit(1)

    if not data or not data.get("status"):
        msg = data.get("message", "Unknown error") if data else "No response"
        print(f"ERROR: Login failed — {msg}")
        sys.exit(1)

    session = data["data"]
    feed_token = smart.getfeedToken()

    return {
        "auth_token": session["jwtToken"],
        "refresh_token": session["refreshToken"],
        "feed_token": feed_token,
        "client_code": client_id,
        "api_key": api_key,
    }


def main():
    print("Angel One SmartAPI — Generating session via TOTP...")
    tokens = generate_session()

    print("\nSession generated successfully.")
    print(f"  Client      : {tokens['client_code']}")
    print(f"  Auth Token  : {tokens['auth_token'][:20]}...")
    print(f"  Feed Token  : {tokens['feed_token'][:20]}...")
    print(f"  Refresh     : {tokens['refresh_token'][:20]}...")
    print("\nTokens are held in memory by the app at startup.")
    print("No .env update needed — TOTP regenerates tokens automatically.")


if __name__ == "__main__":
    main()
