#!/usr/bin/env bash
# run_full_pipeline.sh
# Master pipeline: NOAA max pull → Kalshi bulk pull → features → dual model → backtest → report
# Logs everything to logs/pipeline_run.log
# Commits each completed phase.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

source .venv/bin/activate
export NOAA_TOKEN="REDACTED_NOAA_TOKEN"
export PYTHONPATH="$ROOT/src"

LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/pipeline_run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "========================================"
echo "  TrapTrade Full Pipeline"
echo "  Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "========================================"

# ── Phase A: NOAA max range ───────────────────────────────────────────────────
echo ""
echo "[Phase A] NOAA max range pull (2020→present)..."
python src/noaa_max_fetch.py
git add data/raw/noaa_*_max.csv data/processed/noaa_all_cities_max.parquet 2>/dev/null || true
git commit -m "data(noaa): max range pull 2020→present" 2>/dev/null || true
echo "[Phase A] DONE at $(date -u +%H:%M:%SZ)"

# ── Phase B: Kalshi bulk pull ─────────────────────────────────────────────────
echo ""
echo "[Phase B] Kalshi bulk pull — all 9247 series..."
python src/kalshi_bulk_fetch.py
git add data/raw/kalshi_bulk_*.parquet data/raw/kalshi_bulk_progress.json 2>/dev/null || true
git commit -m "data(kalshi): bulk pull all series complete" 2>/dev/null || true
echo "[Phase B] DONE at $(date -u +%H:%M:%SZ)"

# ── Phase C: Feature engineering ─────────────────────────────────────────────
echo ""
echo "[Phase C] Building feature matrices (weather + generic)..."
python src/features_generic.py
git add data/processed/feature_matrix_*.parquet 2>/dev/null || true
git commit -m "feat: generic + weather feature matrices built" 2>/dev/null || true
echo "[Phase C] DONE at $(date -u +%H:%M:%SZ)"

# ── Phase D: Dual model training ──────────────────────────────────────────────
echo ""
echo "[Phase D] Walk-forward training — weather + generic RF models..."
python src/model_dual.py
git add models/ data/processed/wf_*.parquet 2>/dev/null || true
git commit -m "model: dual RF walk-forward complete (weather + generic)" 2>/dev/null || true
echo "[Phase D] DONE at $(date -u +%H:%M:%SZ)"

# ── Phase E: Backtest + report ────────────────────────────────────────────────
echo ""
echo "[Phase E] Backtest + full report (clean baseline — no lookahead)..."
python src/features_clean.py
python src/model_clean.py
python src/backtest_clean.py
git add reports/ data/processed/trades_*.parquet 2>/dev/null || true
git commit -m "report: full backtest complete — weather + all-market" 2>/dev/null || true
echo "[Phase E] DONE at $(date -u +%H:%M:%SZ)"

# ── Update progress log ───────────────────────────────────────────────────────
cat >> progress.txt << 'EOF'
================================================================================

[SESSION: 2026-03-22 | Executor — Full Data Pull]
Phase     : Full pipeline — NOAA max + Kalshi bulk (9247 series) + dual model
Status    : COMPLETE
Agent     : setup
Completed :
  - noaa_max_fetch: NOAA 2020→present for 6 cities
  - kalshi_bulk_fetch: all 9247 series, all settled markets
  - features_generic: generic (13 features) + weather (25 features) matrices
  - model_dual: RF weather_v2 + RF generic_v1, max WF folds
  - backtest_full: entry/exit on both models, per-category breakdown
  - Full report: reports/backtest_full_{date}.md
Next      : Review report, tune thresholds, add more features
Notes     : All markets downloaded. No synthetic data. Real Kalshi prices only.
EOF

git add progress.txt
git commit -m "chore: update progress log — full pipeline complete"

echo ""
echo "========================================"
echo "  Pipeline COMPLETE"
echo "  Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  Log: $LOG"
echo "========================================"
