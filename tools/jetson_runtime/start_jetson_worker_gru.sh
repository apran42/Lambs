#!/usr/bin/env bash
set -euo pipefail

cd /home/lambs/sheperd_runtime/shepherd_pr7_test/backend
engine='/media/lambs/BEEZAP BZ36/shepherd/engines/best_21videos_20261002_fp16.engine'
if [[ ! -f "$engine" ]]; then
  echo "TensorRT engine is unavailable: $engine" >&2
  exit 1
fi

exec /usr/bin/python3 -m jetson_worker.server \
  --engine "$engine" \
  --cameras jetson_worker/cameras.local.json \
  --host 127.0.0.1 --port 8766 --warmup 1
