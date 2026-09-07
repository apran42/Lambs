# Shepherd-AI implementation roadmap

## Confirmed requirements

- Development sources: `sample_data (10).mp4` and `sample_data (11).mp4`
  (both are 1280 x 720 at 30 FPS and leave more local headroom for two streams)
- Target capture: fixed USB webcam (exact model deferred)
- Target frame format: 640 x 480, at least 20 FPS per displayed stream
- Camera scale: four desired; demonstrate an extensible design and validate at least two
- Test scene: a configurable model zone on foam board with person figures
- Prediction horizon: five minutes
- Jetson target: Jetson Nano 4 GB; JetPack/L4T/CUDA/TensorRT details deferred
- Deployment split: decide after measuring Jetson inference, API, encoding, and database load

## Density phase (initial implementation complete)

Each camera will own a configuration containing:

- camera ID and source
- valid image ROI
- image-to-model-plane homography
- model-zone area in square metres (a configurable reference area)
- calibration version
- density thresholds

The current calculation uses each detection's bottom-centre footpoint. The four
ROI corners are mapped to the virtual model plane by a homography. It reports:

- whole ROI density: `ROI people / (zone width x zone height)`
- fixed cell density: `people in cell / cell area`
- local peak density: the largest `people in movable window / window area`

The local-peak algorithm remains implemented, but it is currently disabled by
configuration. Safety status and five-minute forecasting use the densest fixed
grid. It can be re-enabled later with `ENABLE_LOCAL_PEAK_DENSITY=true` after the
team agrees on how a movable area should be presented and validated.

The planned outdoor density levels are:

- Relaxed: density <= 2.0 people/m2
- Caution: 2.0 < density < 5.0 people/m2
- Danger: density >= 5.0 people/m2

These values follow the project interpretation of the Ministry of the Interior
and Safety [crowd-safety guideline](https://www.mois.go.kr/frt/bbs/type001/commonSelectBoardArticle.do?bbsId=BBSMSTR_000000000015&nttId=121405).
The UI and API must display that this is a model-zone estimate, not a certified
real-world safety measurement.

## Prediction phase (online baseline implemented)

Collect trustworthy time-series observations before training a prediction model.
At minimum, store camera/zone IDs, UTC timestamp, ROI count, density, density
level, inference latency, capture FPS, inference FPS, and skipped-frame count.
Use a time-ordered split and compare a simple seasonal or moving-average baseline
before adopting a more complex model. The output horizon is five minutes.

The current runtime provides a conservative baseline: persistence during the
first minute, followed by a damped linear trend over five-second median buckets.
It exposes readiness and confidence, but its accuracy cannot be claimed until
representative ground-truth time-series data is collected and backtested.

## Multi-camera acceptance

Use one capture runtime per camera and one shared batch inference scheduler.
Before claiming production two-camera support, measure two simultaneous 640 x
480, 20 FPS inputs for at least 30 minutes and report capture FPS, displayed FPS,
inference FPS, latency percentiles, skipped frames, memory, CPU/GPU load, and
temperature. "No frame drop" must be defined separately for capture/display and
AI inference because a latest-frame real-time pipeline intentionally skips stale
frames when inference is slower than capture.

## Deferred decisions

- Exact JetPack, L4T, CUDA, TensorRT, Python, and Ultralytics versions
- TensorRT FP16 versus INT8 engine
- USB camera index, codec, and reconnect behaviour
- All-in-one Jetson deployment versus edge inference plus central API/database
- Ground-truth annotation format and accuracy targets
