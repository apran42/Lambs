# Jetson TensorRT Worker

이 폴더에는 Python 3.6과 호환되는 TensorRT Worker가 있습니다. JetPack의 CUDA
Runtime을 `ctypes`로 직접 사용하므로 PyCUDA를 설치하거나 시스템 패키지를
변경할 필요가 없습니다.

정적 배치 크기 1인 엔진을 지원하며, 활성화된 입력 소스마다 별도 캡처 스레드를
사용합니다. 하나의 TensorRT 추론 스레드가 카메라를 순환하며 처리합니다. 큐에는
항상 최신 프레임만 유지하므로 오래된 프레임이 쌓이지 않습니다. 카메라 원본 시점의
JPEG와 정규화된 검출 결과를 localhost HTTP로 Python 3.8 FastAPI에 전송합니다.

## 외장 드라이브 사전 점검

TensorRT 엔진과 개발용 영상은 외장 드라이브에 저장되어 있습니다. Worker를
실행하기 전에 항상 마운트 상태를 확인하세요. 연결이 끊겨도 빈 디렉터리는 남을 수
있으므로 `test -d`만으로는 확인할 수 없습니다. `mountpoint`와 실제 파일 검사를
함께 사용합니다.

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

`DRIVE MOUNTED`와 `ENGINE OK`가 모두 출력되어야 Worker를 실행할 수 있습니다.
카메라 설정의 파일 입력 경로와 JSON 문법도 확인합니다.

```bash
python3 -m json.tool jetson_worker/cameras.local.json
grep '"source"' jetson_worker/cameras.local.json
```

`0`과 같은 숫자 USB 카메라 입력은 파일이 아니므로 다음 명령으로 확인합니다.

```bash
v4l2-ctl --list-devices
ls -l /dev/video*
```

## 예기치 않은 연결 해제 복구

Worker 실행 중 드라이브가 빠지거나 마운트가 해제되면 FastAPI는 계속 접속될 수
있지만 `/health`에는 카메라 오류 또는 Worker 패킷 대기 상태가 표시됩니다. 다음
순서로 복구하세요.

1. Worker 터미널에서 `Ctrl+C`를 누릅니다. 터미널을 찾을 수 없다면 Worker
   프로세스만 확인한 뒤 해당 PID를 종료합니다.

   ```bash
   pgrep -af 'jetson_worker.server'
   kill <WORKER_PID>
   ```

2. 드라이브를 다시 연결하고 파티션 이름을 확인합니다. 항상 `/dev/sda1`이라고
   가정하면 안 됩니다.

   ```bash
   lsblk -f
   ```

3. 데스크톱 마운트 서비스가 있다면 다음 방법을 우선 사용합니다. `/dev/sda1`은
   `lsblk`에서 확인한 실제 파티션으로 바꿉니다.

   ```bash
   udisksctl mount -b /dev/sda1
   ```

4. `udisksctl`을 사용할 수 없다면 수동으로 마운트합니다.

   ```bash
   sudo mkdir -p "/media/lambs/BEEZAP BZ36"
   sudo mount /dev/sda1 "/media/lambs/BEEZAP BZ36"
   ```

5. 위의 사전 점검을 다시 수행합니다. 마운트 위치가 바뀌었다면 Worker의
   `--engine` 경로와 `jetson_worker/cameras.local.json`의 모든 파일 `source`
   경로를 수정합니다.

6. Worker를 다시 실행한 뒤 Worker와 FastAPI 상태를 차례로 확인합니다.

   ```bash
   curl http://127.0.0.1:8766/health
   curl http://127.0.0.1:8001/health
   ```

마운트에 실패하면 최근 커널 메시지를 확인합니다.

```bash
dmesg | tail -n 30
```

파일시스템 검사는 파티션이 마운트되지 않은 상태에서만 실행하세요. 먼저 읽기 전용
검사를 수행하고, 외장 데이터를 백업하기 전에는 자동 복구를 적용하지 마세요.
구형 JetPack에서 `fsck.exfat`이 없다면 `exfatfsck`를 사용합니다.

```bash
mountpoint -q "/media/lambs/BEEZAP BZ36" || sudo fsck.exfat -n /dev/sda1
# 구형 환경 대체 명령
mountpoint -q "/media/lambs/BEEZAP BZ36" || sudo exfatfsck -n /dev/sda1
```

연결 해제 시 `target is busy`가 나오면 강제로 해제하지 말고 드라이브를 사용하는
프로세스를 찾아 정상 종료합니다.

```bash
sudo fuser -vm "/media/lambs/BEEZAP BZ36"
```

계획적으로 분리할 때는 Worker를 먼저 종료하고 안전하게 마운트를 해제합니다.

```bash
udisksctl unmount -b /dev/sda1
```

## Worker 실행

`cameras.example.json`을 Git에 포함되지 않는 로컬 파일로 복사하고, 영상 경로나
숫자 USB 카메라 입력을 설정합니다. 이미 설정한 `cameras.local.json`이 있다면
다시 복사하지 마세요.

`backend` 폴더에서 다음을 실행합니다.

```bash
/usr/bin/python3 -m jetson_worker.server \
  --engine="/media/lambs/BEEZAP BZ36/shepherd/engines/best_fp16.engine" \
  --cameras=jetson_worker/cameras.local.json \
  --host=127.0.0.1 --port=8766
```

로컬 상태 주소는 `http://127.0.0.1:8766/health`입니다. FastAPI는
`/api/packet/{camera_id}`를 롱 폴링하고 기존 WebSocket 형식으로 프론트엔드에
전달합니다. 외부 네트워크 접근이 꼭 필요한 경우가 아니면 Worker는 localhost에만
바인딩합니다.

## 단일 이미지 시험

Jetson 시스템 Python 3.6을 사용하며 `backend` 폴더에서 실행합니다.

```bash
/usr/bin/python3 -m jetson_worker.cli \
  --engine="/media/lambs/BEEZAP BZ36/shepherd/engines/best_fp16.engine" \
  --image="/media/lambs/BEEZAP BZ36/shepherd/test/test.jpg" \
  --output-image="/media/lambs/BEEZAP BZ36/shepherd/test/result.jpg"
```

## 짧은 영상 성능 시험

```bash
/usr/bin/python3 -m jetson_worker.cli \
  --engine="/media/lambs/BEEZAP BZ36/shepherd/engines/best_fp16.engine" \
  --video="/media/lambs/BEEZAP BZ36/shepherd/videos/input.mp4" \
  --max-frames=100
```

검출 JSON은 개발용 검출기와 동일한 형식을 사용합니다.

```json
{"box":[10.0,20.0,100.0,200.0],"confidence":0.91,"track_id":null}
```

CLI는 기본적으로 5회 예열 추론을 수행합니다. 첫 예열에는 CUDA 컨텍스트 초기화가
포함되며, 이미지 결과의 `inference_ms`는 이후의 안정 상태 추론 시간을 나타냅니다.
의도적으로 콜드 스타트 시간을 측정할 때만 `--warmup=0`을 사용하세요.
