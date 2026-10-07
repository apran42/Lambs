#!/usr/bin/env bash
set -euo pipefail

cd /home/lambs/sheperd_runtime/shepherd_pr7_test/backend
export FORECAST_MODE=gru
export FORECAST_HORIZON_SECONDS=60
export GRU_MODEL_DIR='/media/lambs/BEEZAP BZ36/shepherd/gru_trial_20261007'

exec /home/lambs/sheperd_runtime/backend_venv38/bin/uvicorn main:app \
  --host 0.0.0.0 --port 8001 --workers 1
