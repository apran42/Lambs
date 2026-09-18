# Shepherd-AI 백엔드

백엔드는 영상 소스마다 하나의 캡처 작업자를 사용하고, 모든 카메라가 하나의
배치 추론 엔진을 공유합니다. 영상 캡처와 인코딩은 추론과 독립적으로 실행되므로
개발 환경의 각 스트림은 최소 20 FPS 표시를 목표로 하며, 밀도 계산에는 가장
최근의 분석 결과를 사용합니다. CPU 전용 개발 PC에서는 추론이 20 FPS에 미치지
못할 수 있으므로 스트림 FPS와 카메라별 AI 분석 FPS를 구분해 제공합니다.

## 로컬 실행

Python 3.10 이상이 필요합니다. 아래 명령은 `backend` 폴더에서 실행합니다.

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

`.env.example`을 `.env`로 복사한 뒤 환경에 맞게 값을 수정합니다. 상대 경로는
Uvicorn을 어느 폴더에서 실행하더라도 `backend` 폴더를 기준으로 해석됩니다.
하드웨어 시험에서 첫 번째 USB 웹캠을 사용하려면 `VIDEO_PATH=0`으로 설정합니다.
CPU 전용 개발 환경의 `CPU_INFERENCE_THREADS`와 `OPENCV_THREADS`는 독립 캡처
작업자에 필요한 처리 자원을 확보하는 설정이며, 향후 CUDA/TensorRT 추론 엔진을
제한하지 않습니다.

## 밀도 계산

각 사람 바운딩박스의 아래쪽 중앙점(발 위치)을 사용합니다. 정규화된 ROI 꼭짓점
4개를 호모그래피로 설정 가능한 모델 평면에 투영한 뒤 다음 값을 계산합니다.

- 전체 ROI 밀도
- 고정 그리드 셀별 밀도
- 이동 가능한 국소 영역에서의 최대 밀도

ROI 설정은 `GET/PUT /api/roi`로 조회하거나 변경할 수 있습니다. 기본값은
3m x 3m 가상 구역과 1㎡ 크기의 3 x 3 셀입니다. 국소 최대 밀도 계산 코드는
유지되어 있지만 기본적으로 비활성화되어 있습니다
(`ENABLE_LOCAL_PEAK_DENSITY=false`). 현재 위험 단계는 가장 밀도가 높은 고정
그리드 셀을 기준으로 판정합니다.

## 5분 예측

분석된 프레임마다 5분 뒤 예측값을 갱신합니다. 시작 후 첫 1분 동안은
`ready=false`인 현재값 유지 기준선을 반환하고, 충분한 이력이 쌓이면 5초 단위
중앙값 버킷에 감쇠 선형 추세를 적용합니다. 최신 결과는
`/api/forecast/{camera_id}` 또는 WebSocket 메타데이터의 `forecast_5m`에서
확인할 수 있습니다. 대표성 있는 시계열 정답 데이터를 수집한 후에는 이 온라인
기준선을 학습 모델로 교체하거나 정확도를 검증해야 합니다.

## 검증

```bash
python -m compileall -q .
python -c "from tests.test_stream_service import test_clients_share_one_encoded_packet; test_clients_share_one_encoded_packet()"
```

전체 테스트를 실행하려면 `requirements-dev.txt`를 설치한 뒤 `pytest`를 실행합니다.

## WebSocket 패킷 형식

패킷의 첫 4바이트는 빅엔디언 부호 없는 정수로 표현한 JSON 길이입니다. 그 뒤에
UTF-8 JSON 메타데이터와 JPEG 바이트가 이어집니다. 프로토콜 버전 1에는 카메라
ID, 프레임 ID, 시각, 영상 크기, 인원수, 임시 상태와 검출 결과가 포함됩니다.
각 검출 결과에는 `box`, `track_id`, `confidence`가 들어갑니다.

`cameras.json`에는 현재 개발용 영상 2개가 활성화되어 있습니다. 모델 인스턴스를
추가로 만들지 말고 이 파일에 카메라를 추가하세요. 스트림 주소는
`/ws/stream/{camera_id}`이며 `/api/cameras`에서 캡처 상태와 FPS를 확인할 수
있습니다. 프로세스가 모든 캡처와 공유 추론 엔진을 소유하므로 Uvicorn 작업자는
반드시 하나만 사용합니다. TensorRT 배포는 JetPack 환경이 확정된 후 진행합니다.
