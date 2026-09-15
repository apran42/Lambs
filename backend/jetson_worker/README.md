# Jetson TensorRT worker

This directory contains the Python 3.6-compatible TensorRT worker. It
uses the CUDA Runtime from JetPack through `ctypes`, so PyCUDA is not required
and no system package needs to be modified.

The worker supports a static batch-one engine, captures each enabled source on
its own thread, and shares one TensorRT inference thread in round-robin order.
Only the newest frame is retained; old frames never accumulate in a queue. The
worker sends camera-perspective JPEG frames and normalized detections to the
Python 3.8 FastAPI process over localhost HTTP.

## Persistent service

Copy `cameras.example.json` to an untracked local file, set the video or numeric
USB camera sources, then run:

```bash
/usr/bin/python3 -m jetson_worker.server \
  --engine="/media/lambs/BEEZAP BZ36/shepherd/engines/best_fp16.engine" \
  --cameras=jetson_worker/cameras.local.json \
  --host=127.0.0.1 --port=8766
```

Check `http://127.0.0.1:8766/health` locally. The FastAPI bridge long-polls
`/api/packet/{camera_id}` and exposes the existing WebSocket protocol to the
frontend. Keep the worker bound to localhost unless network access is explicitly
required.

## Smoke tests

Run from the backend directory with Jetson system Python 3.6:

```bash
/usr/bin/python3 -m jetson_worker.cli \
  --engine="/media/lambs/BEEZAP BZ36/shepherd/engines/best_fp16.engine" \
  --image="/media/lambs/BEEZAP BZ36/shepherd/test/test.jpg" \
  --output-image="/media/lambs/BEEZAP BZ36/shepherd/test/result.jpg"
```

For a short video benchmark:

```bash
/usr/bin/python3 -m jetson_worker.cli \
  --engine="/media/lambs/BEEZAP BZ36/shepherd/engines/best_fp16.engine" \
  --video="/media/lambs/BEEZAP BZ36/shepherd/videos/input.mp4" \
  --max-frames=100
```

The JSON detections use the same contract as the development detector:

```json
{"box":[10.0,20.0,100.0,200.0],"confidence":0.91,"track_id":null}
```

The CLI performs five warm-up inferences by default. The first warm-up includes
CUDA context initialization, while the reported image `inference_ms` represents
a subsequent steady-state inference. Use `--warmup=0` only when measuring cold
startup deliberately.
