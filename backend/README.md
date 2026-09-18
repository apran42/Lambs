# Shepherd-AI 백엔드

백엔드는 영상 소스마다 하나의 캡처 작업자를 사용하고 모든 카메라가 하나의 배치
추론 엔진을 공유합니다. 캡처·인코딩은 추론과 독립적으로 실행됩니다. 개발 환경의
각 스트림은 최소 20 FPS 표시를 목표로 하며 밀도 계산에는 최신 분석 결과를
사용합니다. CPU 전용 PC에서는 추론이 20 FPS에 미치지 못할 수 있으므로 스트림
FPS와 카메라별 AI 분석 FPS를 구분해 제공합니다.

## 로컬 개발 환경 실행

Python 3.10 이상이 필요합니다. 아래 명령은 `backend` 폴더에서 실행합니다.

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

`.env.example`을 `.env`로 복사하고 환경에 맞게 수정합니다. 상대 경로는 Uvicorn을
어느 폴더에서 실행하더라도 `backend` 폴더를 기준으로 해석됩니다. 첫 번째 USB
웹캠을 사용하려면 `VIDEO_PATH=0`으로 설정합니다. `CPU_INFERENCE_THREADS`와
`OPENCV_THREADS`는 CPU 개발 환경에서 캡처 작업자의 처리 자원을 확보하며 향후
CUDA/TensorRT 엔진을 제한하지 않습니다.

## Jetson Nano API 전용 실행

Jetson에서는 Python 3.8 FastAPI와 Python 3.6 TensorRT Worker를 분리합니다.
`requirements-jetson-backend.txt`는 격리된
`~/sheperd_runtime/backend_venv38` 환경에만 설치하세요. 이 목록에는 torch,
Ultralytics, TensorRT, CUDA, OpenCV가 포함되지 않습니다. 시스템 Python이나
기존 `~/backend_venv`에는 설치하지 마세요.

TensorRT Worker가 영상 캡처, JPEG 인코딩과 추론을 담당합니다.
`jetson_worker/cameras.example.json`을 로컬 설정 파일로 복사하고 외장 드라이브의
영상 경로를 수정하세요. 두 프로세스가 동일한 파일을 사용해야 활성 카메라 ID가
일치합니다. 시작 전에
[`jetson_worker/README.md`](jetson_worker/README.md#외장-드라이브-사전-점검)의
외장 드라이브 점검 및 복구 절차를 수행하세요.

### 터미널 1: TensorRT Worker

```bash
cd ~/sheperd_runtime/shepherd_backend/backend
cp jetson_worker/cameras.example.json jetson_worker/cameras.local.json
/usr/bin/python3 -m jetson_worker.server \
  --engine "/media/lambs/BEEZAP BZ36/shepherd/engines/best_fp16.engine" \
  --cameras jetson_worker/cameras.local.json \
  --host 127.0.0.1 --port 8766
```

`cameras.local.json`이 이미 올바르게 설정되어 있다면 다시 복사하지 마세요. 복사하면
기존 로컬 설정을 덮어쓸 수 있습니다.

### 터미널 2: FastAPI

`.env.jetson-backend.example`을 `.env`로 복사한 뒤 Python 3.8 환경에서 실행합니다.

```bash
cd ~/sheperd_runtime/shepherd_backend/backend
source ~/sheperd_runtime/backend_venv38/bin/activate
pip install -r requirements-jetson-backend.txt
export INFERENCE_BACKEND=jetson
export JETSON_WORKER_URL=http://127.0.0.1:8766
python -c "import main"
uvicorn main:app --host 0.0.0.0 --port 8001 --workers 1
```

두 프로세스 모두 하나의 작업자만 사용합니다. 상태 확인 명령은 다음과 같습니다.

```bash
curl http://127.0.0.1:8766/health
curl http://127.0.0.1:8001/health
```

FastAPI의 `/health`는 `mode: jetson-worker-bridge`를 반환하며 첫 패킷을 받은 뒤
`inference.available`이 `true`로 바뀝니다. FastAPI 프로세스는 OpenCV,
TensorRT, PyCUDA, torch, Ultralytics를 사용하지 않습니다. Worker가 전송한
정규화 검출 결과로 ROI 밀도와 5분 예측을 계산합니다. 개발 PC는
`INFERENCE_BACKEND=ultralytics`를 사용하되 동일한 검출 형식(`box`,
`confidence`, 선택 항목 `track_id`)을 유지합니다.

## 밀도 및 예측

밀도 계산에는 사람 바운딩박스의 아래쪽 중앙점(발 위치)을 사용합니다. ROI 꼭짓점
4개를 호모그래피로 모델 평면에 투영하여 전체 ROI, 고정 그리드 셀, 이동 국소
영역의 밀도를 계산합니다. `GET/PUT /api/roi`로 설정을 변경할 수 있습니다.
기본값은 3m x 3m 가상 구역과 1㎡ 크기의 3 x 3 셀입니다. 이동 국소 밀도는
기본적으로 비활성화되어 있으며(`ENABLE_LOCAL_PEAK_DENSITY=false`), 위험 단계는
가장 밀도가 높은 고정 셀을 기준으로 합니다.

5분 예측은 시작 후 첫 1분 동안 현재값 유지 기준선을 사용하고, 이후 5초 단위
중앙값에 감쇠 선형 추세를 적용합니다. `/api/forecast/{camera_id}` 또는 WebSocket
메타데이터의 `forecast_5m`에서 확인할 수 있습니다. 대표성 있는 시계열 정답을
수집한 뒤 학습 모델로 교체하거나 정확도를 검증해야 합니다.

## 검증

```bash
python -m compileall -q .
python -c "from tests.test_stream_service import test_clients_share_one_encoded_packet; test_clients_share_one_encoded_packet()"
```

전체 테스트는 `requirements-dev.txt`를 설치한 뒤 `pytest`로 실행합니다.

## WebSocket 패킷 형식

첫 4바이트는 빅엔디언 부호 없는 정수로 표현한 JSON 길이이며, UTF-8 JSON
메타데이터와 JPEG 바이트가 이어집니다. 프로토콜 버전 1은 카메라/프레임 ID,
시각, 영상 크기, 인원수, 임시 상태와 검출 결과를 포함합니다. 각 검출 결과에는
`box`, `track_id`, `confidence`가 들어갑니다.

스트림 주소는 `/ws/stream/{camera_id}`이고 `/api/cameras`에서 캡처 상태와 FPS를
확인할 수 있습니다. 카메라는 별도 모델 인스턴스를 만들지 말고 설정 파일에
추가하세요.
