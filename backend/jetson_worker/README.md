# Jetson TensorRT worker smoke test

This directory contains the first Python 3.6-compatible TensorRT worker. It
uses the CUDA Runtime from JetPack through `ctypes`, so PyCUDA is not required
and no system package needs to be modified.

The first milestone intentionally supports a static batch-one engine and emits
normalized detection dictionaries. Tracking, GStreamer capture, multi-camera
scheduling, and FastAPI IPC are later milestones.

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

