#!/usr/bin/env bash
# init.sh — TrapTrade environment setup, dependency install, smoke test
# Run: bash init.sh
# Backtest only. No live execution.

set -euo pipefail

echo "=============================="
echo "  TrapTrade — Init Script"
echo "=============================="

# ── 1. ENV VARS ──────────────────────────────────────────────────────────────
: "${NOAA_TOKEN:?NOAA_TOKEN not set. Copy .env.example to .env, fill in NOAA_TOKEN, then run: set -a; source .env; set +a}"
export NOAA_BASE_URL="https://www.ncdc.noaa.gov/cdo-web/api/v2"
export POLYMARKET_GAMMA_URL="https://gamma-api.polymarket.com"

# Telegram (FUTURE — leave empty until alerting_telegram layer is unlocked)
export TELEGRAM_TOKEN=""
export TELEGRAM_CHAT_ID=""

echo "[env] Environment variables set."

# ── 2. PYTHON VERSION CHECK ───────────────────────────────────────────────────
PYTHON_BIN="python3"
PYTHON_VERSION=$($PYTHON_BIN --version 2>&1 | awk '{print $2}')
echo "[python] Found: $PYTHON_VERSION"

# ── 3. VIRTUALENV ─────────────────────────────────────────────────────────────
VENV_DIR="$(pwd)/.venv"
if [ ! -d "$VENV_DIR" ]; then
  echo "[venv] Creating virtualenv at .venv ..."
  $PYTHON_BIN -m venv "$VENV_DIR"
else
  echo "[venv] .venv already exists, skipping creation."
fi

# Use venv binaries for all subsequent commands
PIP="$VENV_DIR/bin/pip"
PYTHON="$VENV_DIR/bin/python"

echo "[venv] Activated: $VENV_DIR"

# ── 4. DIRECTORY STRUCTURE ───────────────────────────────────────────────────
echo "[dirs] Creating project directories..."
mkdir -p data/raw data/processed models reports docs

echo "[dirs] Structure:"
echo "  data/raw/          — raw NOAA + Polymarket downloads"
echo "  data/processed/    — cleaned, joined datasets"
echo "  models/            — trained RF models (.joblib)"
echo "  reports/           — backtest reports (.md, .csv)"
echo "  docs/              — audit, notes"

# ── 5. INSTALL DEPENDENCIES ──────────────────────────────────────────────────
echo "[deps] Installing Python dependencies into .venv ..."

$PIP install --quiet --upgrade pip

$PIP install --quiet \
  "scikit-learn==1.8.0" \
  "pandas==3.0.0" \
  "numpy==2.4.1" \
  "requests==2.32.5" \
  "joblib==1.5.3" \
  "python-telegram-bot==22.7" \
  "pyarrow>=16.0"

echo "[deps] All dependencies installed."

# ── 6. SMOKE TEST ─────────────────────────────────────────────────────────────
echo "[smoke] Running smoke tests..."

$PYTHON - <<'PYEOF'
import sys

# Core imports
import sklearn
import pandas as pd
import numpy as np
import requests
import joblib

print(f"  scikit-learn : {sklearn.__version__}")
print(f"  pandas       : {pd.__version__}")
print(f"  numpy        : {np.__version__}")
print(f"  requests     : {requests.__version__}")
print(f"  joblib       : {joblib.__version__}")

# Env vars check
import os
token = os.environ.get("NOAA_TOKEN", "")
if not token:
    print("  [WARN] NOAA_TOKEN not set")
else:
    print(f"  NOAA_TOKEN   : {'*' * (len(token) - 4)}{token[-4:]}")

# NOAA API connectivity (HEAD request only, no data pulled)
try:
    resp = requests.head(
        "https://www.ncdc.noaa.gov/cdo-web/api/v2/datatypes",
        headers={"token": token},
        timeout=5
    )
    print(f"  NOAA API     : reachable (status {resp.status_code})")
except Exception as e:
    print(f"  NOAA API     : [WARN] {e}")

print("\n[smoke] All checks passed.")
PYEOF

echo ""
echo "=============================="
echo "  Init complete. Ready for"
echo "  Fase 2: Environment Audit."
echo "=============================="
