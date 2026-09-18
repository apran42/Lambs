# Jetson Nano InfluxDB 마이그레이션 안내

이 문서는 개발 PC의 InfluxDB 적재를 Jetson Nano로 이전하고, 네트워크나 DB가
일시적으로 중단되어도 측정값을 잃지 않도록 구성하는 절차를 설명합니다.

## 목표 구조

```text
카메라
  → TensorRT Worker(Python 3.6, 127.0.0.1:8766)
  → FastAPI(Python 3.8, 0.0.0.0:8001)
  → SQLite WAL Outbox
  → InfluxDB(127.0.0.1:8086)
```

FastAPI는 측정값을 먼저 SQLite Outbox에 기록합니다. 별도 백그라운드 스레드가
Outbox를 InfluxDB로 전송하고, 성공한 행만 삭제합니다. InfluxDB가 중단되면 행은
Outbox에 남고 복구 후 자동으로 재전송됩니다.

## 1. 저장 장치 준비

현재 `BEEZAP BZ36` 외장 파티션은 exFAT입니다. InfluxDB 데이터와 SQLite WAL은
exFAT에 저장하지 마세요. 데이터베이스가 요구하는 파일 잠금과 장애 복구 동작을
보장하기 위해 ext4 파티션 또는 별도의 ext4 SSD가 필요합니다.

파티션 변경이나 포맷 전에 외장 드라이브 전체를 백업하세요. 이 문서에서는 ext4
저장소가 `/mnt/shepherd-data`에 마운트됐다고 가정합니다.

```bash
lsblk -f
findmnt /mnt/shepherd-data
mountpoint -q /mnt/shepherd-data && echo "EXT4 STORAGE MOUNTED"
findmnt -no FSTYPE /mnt/shepherd-data
```

마지막 명령 결과가 `ext4`인지 확인합니다. 애플리케이션은
`INFLUXDB_OUTBOX_MOUNT`가 실제 마운트 지점이 아니면 Outbox를 시작하지 않습니다.
외장 드라이브가 빠진 상태에서 SD 카드의 같은 경로에 데이터가 기록되는 것을 막기
위한 보호 장치입니다.

권장 디렉터리:

```bash
sudo mkdir -p /mnt/shepherd-data/influxdb
sudo mkdir -p /mnt/shepherd-data/outbox
sudo chown -R lambs:lambs /mnt/shepherd-data/outbox
```

InfluxDB 데이터 디렉터리의 소유자는 설치 방식에서 사용하는 InfluxDB 서비스
계정에 맞춰 설정해야 합니다.

## 2. InfluxDB 설치 및 초기 설정

JetPack/L4T의 Ubuntu 버전과 CPU 아키텍처에 맞는 InfluxDB 패키지를 사용합니다.
설치 전에 다음 값을 기록하고, 팀 전체가 같은 InfluxDB 주 버전을 사용하세요.

```bash
uname -m
lsb_release -a
```

InfluxDB 초기 설정값:

```text
조직: Lambs
버킷: crowd_monitor
수집 주기: 카메라당 1초
접속 주소: http://127.0.0.1:8086
```

토큰은 Git에 커밋하지 말고 Jetson의 `backend/.env`에만 저장합니다. InfluxDB의
실제 데이터 디렉터리는 `/mnt/shepherd-data/influxdb`를 사용하도록 설치된 버전의
공식 설정 방법에 따라 변경합니다.

## 3. FastAPI 환경변수

`backend/.env.jetson-backend.example`을 `backend/.env`로 복사하고 토큰을
교체합니다.

```env
INFLUXDB_ENABLED=true
INFLUXDB_URL=http://127.0.0.1:8086
INFLUXDB_TOKEN=실제_토큰
INFLUXDB_ORG=Lambs
INFLUXDB_BUCKET=crowd_monitor
INFLUXDB_OUTBOX_PATH=/mnt/shepherd-data/outbox/metrics_outbox.sqlite3
INFLUXDB_OUTBOX_MOUNT=/mnt/shepherd-data
INFLUXDB_RETRY_SECONDS=5
INFLUXDB_BATCH_SIZE=100
DB_WRITE_INTERVAL_SECONDS=1
```

Outbox에는 토큰이 저장되지 않습니다. 카메라 ID, 구역, 인원수, 밀도와 측정 시각만
저장됩니다.

## 4. 실행 순서

1. ext4 저장소 마운트 확인
2. InfluxDB 실행 및 상태 확인
3. TensorRT Worker 실행
4. FastAPI 실행

```bash
curl http://127.0.0.1:8086/health
curl http://127.0.0.1:8766/health
curl http://127.0.0.1:8001/health
```

FastAPI `/health`의 `storage` 항목을 확인합니다.

```json
{
  "storage": {
    "enabled": true,
    "available": true,
    "pending_rows": 0,
    "last_success_at": "2026-09-18T10:00:00+00:00",
    "last_error": null
  }
}
```

- `available=false`, `pending_rows>0`: InfluxDB 전송 실패, Outbox에는 보관 중
- `available=true`, `pending_rows=0`: 정상 적재
- `enabled=false`: 환경변수 또는 필수 마운트 상태 확인 필요

## 5. 기존 데이터 이전

개발 PC와 Jetson이 동일한 InfluxDB 주 버전을 사용할 때 공식 `influx backup`과
`influx restore` 명령을 사용합니다.

1. 개발 PC의 백엔드 쓰기를 중단합니다.
2. 개발 PC InfluxDB를 백업합니다.
3. 백업 폴더를 Jetson으로 복사합니다.
4. Jetson InfluxDB에 복원합니다.
5. 최근 시각과 레코드 개수를 비교합니다.
6. Jetson FastAPI 쓰기를 활성화합니다.

버전이 다르면 Line Protocol 또는 CSV 내보내기·가져오기를 사용합니다. 마이그레이션
동안 개발 PC와 Jetson이 동시에 같은 시각의 데이터를 기록하지 않도록 쓰기 전환
시각을 기록하세요.

## 6. 장애 시험

운영 전 다음 항목을 확인합니다.

1. InfluxDB를 2분간 중지했을 때 `pending_rows`가 증가하는지 확인
2. InfluxDB 재시작 후 `pending_rows`가 0으로 감소하는지 확인
3. 같은 카메라·측정 시각 데이터가 중복되지 않는지 확인
4. FastAPI 재시작 후 Outbox가 유지되는지 확인
5. 카메라 2대를 30분 이상 실행해 FPS, RAM, SWAP, 온도 측정

```bash
tegrastats
```

InfluxDB 추가 후 화면 표시 FPS나 TensorRT 분석 FPS가 지속적으로 떨어지거나
SWAP 사용량이 계속 증가하면 InfluxDB를 중앙 서버로 옮기고 Jetson의 Outbox는
그대로 유지합니다. 이 경우 `INFLUXDB_URL`만 중앙 서버 주소로 변경하면 됩니다.

## 7. 데이터 정합성 기준

- 시계열 기준 시각은 DB 도착 시각이 아니라 프레임의 `captured_at` 또는
  `analysis_captured_at`을 사용합니다.
- Outbox 고유 키는 `camera_id + 측정 시각`입니다.
- 같은 측정값을 재전송해도 동일한 InfluxDB 시계열 지점에 기록됩니다.
- 예측 모델은 도착 순서가 아니라 측정 시각 순서로 입력을 정렬해야 합니다.
- FastAPI 재시작 후 예측 모델은 최근 이력을 DB에서 다시 읽어 상태를 복원해야
  합니다. 예측 상태 복원은 후속 구현 항목입니다.
