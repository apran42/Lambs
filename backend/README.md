# Shepherd-AI backend

The backend uses one capture worker per video source and one shared batch
inference engine. Video capture/encoding remains independent from inference, so
each development stream targets at least 20 FPS while density uses the newest
available analysis. Stream FPS and per-camera AI analysis FPS are reported
separately because a CPU-only development machine may not infer at 20 FPS.

## Local run

Python 3.10 or newer is required.

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

Copy `.env.example` to `.env` and adjust the values. Relative paths are resolved
from the `backend` directory, regardless of the directory used to launch Uvicorn.
Set `VIDEO_PATH=0` to select the first USB webcam when hardware testing begins.
For CPU-only development, `CPU_INFERENCE_THREADS` and `OPENCV_THREADS` reserve
processing capacity for the independent capture workers. They do not limit a
future CUDA/TensorRT inference engine.

Density uses the bottom-centre footpoint of each person box. Four normalized ROI
corners are mapped by a homography to the configurable model plane. The backend
reports whole-ROI density, fixed-grid cell density, and the maximum density from
a sliding local-area window. Edit the configuration through `GET/PUT /api/roi`;
the default is a 3 m x 3 m virtual zone divided into 3 x 3 one-square-metre cells.
The sliding local-area implementation remains in the code but is inactive by
default (`ENABLE_LOCAL_PEAK_DENSITY=false`); risk uses the densest fixed grid.

Each analyzed frame also updates a five-minute forecast. During the first minute
it returns a persistence baseline with `ready=false`; after enough history has
accumulated it uses a damped linear trend over five-second median buckets. Read
the latest result from `/api/forecast/{camera_id}` or `forecast_5m` in WebSocket
metadata. This online baseline should be replaced or validated with a trained
time-series model once representative historical and ground-truth data exists.

## Verification

```bash
python -m compileall -q .
python -c "from tests.test_stream_service import test_clients_share_one_encoded_packet; test_clients_share_one_encoded_packet()"
```

For the full test runner, install `requirements-dev.txt` and run `pytest`.

## WebSocket packet

The first four bytes are the unsigned big-endian JSON length, followed by UTF-8
JSON metadata and JPEG bytes. Protocol version 1 includes the camera and frame
IDs, timestamps, dimensions, count, temporary status, and detections. Each
detection contains `box`, `track_id`, and `confidence`.

`cameras.json` currently enables two development videos. Add cameras there rather
than constructing additional model instances. Streams are available at
`/ws/stream/{camera_id}` and `/api/cameras` reports capture health and FPS. Use
one Uvicorn worker because the process owns all captures and the shared inference
engine. TensorRT deployment is intentionally deferred until the JetPack stack is
known.
