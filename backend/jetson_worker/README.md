# Jetson TensorRT worker

This directory contains the Python 3.6-compatible TensorRT worker. It
uses the CUDA Runtime from JetPack through `ctypes`, so PyCUDA is not required
and no system package needs to be modified.

The worker supports a static batch-one engine, captures each enabled source on
its own thread, and shares one TensorRT inference thread in round-robin order.
Only the newest frame is retained; old frames never accumulate in a queue. The
worker sends camera-perspective JPEG frames and normalized detections to the
Python 3.8 FastAPI process over localhost HTTP.

## External drive preflight

The TensorRT engine and development videos are stored on the external drive.
Check the drive before every worker start. A directory can still exist after a
disconnect, so checking only with `test -d` is not sufficient; use `mountpoint`
and then check the actual files.

```bash
DRIVE_MOUNT="/media/lambs/BEEZAP BZ36"
ENGINE_PATH="$DRIVE_MOUNT/shepherd/engines/best_fp16.engine"

lsblk -o NAME,SIZE,FSTYPE,MOUNTPOINT,MODEL
mountpoint -q "$DRIVE_MOUNT" \
  && echo "DRIVE MOUNTED" \
  || echo "DRIVE NOT MOUNTED"
test -r "$ENGINE_PATH" \
  && echo "ENGINE OK: $ENGINE_PATH" \
  || echo "ENGINE NOT FOUND: $ENGINE_PATH"
```

Do not start the worker unless both `DRIVE MOUNTED` and `ENGINE OK` are
reported. Also verify that every file source in the camera configuration exists:

```bash
python3 -m json.tool jetson_worker/cameras.local.json
grep '"source"' jetson_worker/cameras.local.json
```

Numeric USB camera sources such as `0` do not correspond to files and should be
checked with `v4l2-ctl --list-devices` or `ls -l /dev/video*` instead.

## Recovery after an unexpected disconnect

If the drive was unplugged or unmounted while the worker was running, the API
may remain reachable while `/health` reports a camera error or waits for worker
packets. Recover in this order:

1. Stop the TensorRT worker with `Ctrl+C`. If its terminal is gone, locate only
   the worker process and stop that PID:

   ```bash
   pgrep -af 'jetson_worker.server'
   kill <WORKER_PID>
   ```

2. Reconnect the drive and identify its partition. Do not assume that it is
   always `/dev/sda1`:

   ```bash
   lsblk -f
   ```

3. Prefer the desktop mounting service when it is available. Replace
   `/dev/sda1` with the partition reported by `lsblk`:

   ```bash
   udisksctl mount -b /dev/sda1
   ```

4. If `udisksctl` is unavailable, mount it manually:

   ```bash
   sudo mkdir -p "/media/lambs/BEEZAP BZ36"
   sudo mount /dev/sda1 "/media/lambs/BEEZAP BZ36"
   ```

5. Repeat the preflight checks above. If the mount location changed, update
   both the worker `--engine` path and every file `source` in
   `jetson_worker/cameras.local.json` before restarting.

6. Start the TensorRT worker again, confirm its health, and then restart or
   recheck the FastAPI bridge:

   ```bash
   curl http://127.0.0.1:8766/health
   curl http://127.0.0.1:8001/health
   ```

If mounting fails, inspect the latest kernel messages:

```bash
dmesg | tail -n 30
```

Run a filesystem check only while the partition is unmounted. Start with a
read-only check and do not apply repairs until the external data is backed up:

```bash
mountpoint -q "/media/lambs/BEEZAP BZ36" || sudo fsck.exfat -n /dev/sda1
```

If unmounting reports `target is busy`, identify the process using the drive and
stop it normally instead of forcing the unmount:

```bash
sudo fuser -vm "/media/lambs/BEEZAP BZ36"
```

For a planned disconnect, stop the worker first and unmount safely:

```bash
udisksctl unmount -b /dev/sda1
```

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
