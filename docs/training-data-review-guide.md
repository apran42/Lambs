# Lambs 학습 데이터 검수 사이트 사용 안내

이 문서만 따라 하면 영상에서 검수용 프레임을 추출하고, 웹 화면에서 실제
인원수와 사람 바운딩박스를 검수한 뒤 YOLO 학습 데이터로 내보낼 수 있습니다.

검수 결과와 원본 영상은 용량 및 개인정보 문제 때문에 Git에 올리지 않습니다.
팀원별로 서로 다른 영상을 배정하고, 완성된 `exports` 폴더만 담당자에게
전달하는 방식을 권장합니다.

## 1. 준비물

- Windows 10/11 또는 Linux
- Python 3.10~3.12
- Git
- 검수할 MP4 영상
- 자동 박스 후보 생성에 사용할 YOLO `.pt` 모델

NVIDIA GPU는 필수가 아닙니다. GPU가 없으면 CPU로 실행되므로 자동 검출이
느릴 수 있지만 검수 사이트 기능에는 차이가 없습니다.

## 2. 브랜치 받기

```bash
git fetch origin
git switch codex/training-data-review-site
git pull
```

구형 Git에서 `switch`를 지원하지 않으면 다음을 사용합니다.

```bash
git checkout codex/training-data-review-site
git pull
```

## 3. 전용 Python 환경 설치

### Windows PowerShell

프로젝트 최상위 폴더에서 실행합니다.

```powershell
py -3.11 -m venv .venv-training
.\.venv-training\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r tools/requirements-training.txt
```

PowerShell이 스크립트 실행을 막으면 현재 창에서만 다음 명령을 먼저 실행합니다.

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

### Linux

```bash
python3 -m venv .venv-training
source .venv-training/bin/activate
python -m pip install --upgrade pip
python -m pip install -r tools/requirements-training.txt
```

설치 확인:

```bash
python -c "import cv2, fastapi, ultralytics, uvicorn; print('training tools ready')"
```

## 4. 로컬 파일 배치

원본 영상은 `data` 폴더에 놓습니다. `.pt` 모델은 프로젝트 최상위 또는 팀에서
정한 로컬 모델 폴더에 놓습니다. 이 파일들은 Git에서 자동 제외됩니다.

예시:

```text
lambs/
├─ data/
│  └─ sample_data (4).mp4
├─ models/
│  └─ best.pt
└─ tools/
```

모델 파일 경로는 아래 명령의 `--model` 인자에 맞추면 되므로 반드시 위 구조일
필요는 없습니다.

## 5. 검수용 데이터 추출

프로젝트 최상위 폴더에서 실행합니다. 괄호나 공백이 있는 경로는 반드시
따옴표로 감쌉니다.

### Windows PowerShell

```powershell
python tools/extract_training_dataset.py `
  --video "data/sample_data (4).mp4" `
  --model "models/best.pt" `
  --output "data/training/sample_data_4" `
  --sample-fps 0.5 `
  --confidence 0.20 `
  --image-size 640 `
  --batch-size 8 `
  --camera-id cam-01 `
  --zone-id zone-a
```

### Linux

```bash
python tools/extract_training_dataset.py \
  --video "data/sample_data (4).mp4" \
  --model "models/best.pt" \
  --output "data/training/sample_data_4" \
  --sample-fps 0.5 \
  --confidence 0.20 \
  --image-size 640 \
  --batch-size 8 \
  --camera-id cam-01 \
  --zone-id zone-a
```

주요 옵션:

| 옵션 | 의미 | 권장값 |
|---|---|---|
| `--sample-fps` | 영상 1초당 추출할 프레임 수 | 장면 변화가 적으면 `0.5`, 크면 `1.0` |
| `--confidence` | 자동 후보 박스 생성 임계값 | 누락 확인 목적이면 `0.20~0.25` |
| `--image-size` | 자동 검출 입력 크기 | `640` |
| `--batch-size` | 한 번에 추론할 이미지 수 | CPU `4~8`, GPU `8~32` |

출력 폴더가 비어 있지 않으면 기존 검수 결과 보호를 위해 추출이 중단됩니다.
같은 영상을 다시 추출하려면 기존 폴더를 삭제하지 말고 새 출력 이름을
사용하세요.

자동 박스는 검수 편의를 위한 후보일 뿐 정답이 아닙니다.
`labels_auto_yolo`를 검수 없이 학습에 사용하면 안 됩니다.

## 6. 검수 서버 실행

### Windows PowerShell 권장 방법

```powershell
powershell -ExecutionPolicy Bypass -File tools/start_training_data_site.ps1 `
  -Dataset "data/training/sample_data_4" `
  -Port 8765
```

### 모든 운영체제 공통 방법

```bash
python tools/review_training_dataset.py \
  --dataset "data/training/sample_data_4" \
  --host 127.0.0.1 \
  --port 8765
```

브라우저에서 다음 주소를 엽니다.

```text
http://127.0.0.1:8765/
```

서버를 종료하려면 서버를 실행한 터미널에서 `Ctrl+C`를 누릅니다. 다른
데이터셋으로 바꾸려면 반드시 기존 서버를 종료한 뒤 `-Dataset` 또는
`--dataset` 값만 변경하여 다시 실행합니다.

`주소를 이미 사용 중` 오류가 나면 기존 서버가 실행 중인 것입니다. 기존
터미널에서 종료하거나 다른 포트(예: `8766`)를 사용합니다.

### 같은 PC가 아닌 팀원에게 임시 공개

동일한 신뢰 가능한 LAN 안에서만 다음처럼 실행할 수 있습니다.

```powershell
powershell -ExecutionPolicy Bypass -File tools/start_training_data_site.ps1 `
  -Dataset "data/training/sample_data_4" `
  -Port 8765 `
  -HostAddress 0.0.0.0
```

이 사이트에는 로그인 기능이 없으므로 공용 인터넷에 노출하면 안 됩니다.
여러 사람이 동일 프레임을 동시에 저장하면 마지막 저장 내용이 적용되므로,
공동 작업 시 영상 또는 프레임 범위를 팀원별로 명확하게 나누세요.

## 7. 프레임 검수 기준

모든 정상 프레임은 다음 조건을 만족해야 합니다.

```text
실제 인원수 = 삭제하지 않은 활성 바운딩박스 수
```

- `유지(K)`: 실제 사람이고 박스가 충분히 정확함
- `삭제(X)`: 볼라드·표지판·배경 등을 사람으로 오검출함
- `다시 그리기(R)`: 실제 사람이지만 박스 범위가 부정확함
- `누락 박스 추가(N)`: 실제 사람이 있는데 자동 박스가 없음
- `발 위치(F)`: 밀도 계산용 바닥 접점을 지정함
- `가려짐`: 다른 사람이나 물체 때문에 신체 일부가 보이지 않음
- `잘림`: 사람 일부가 영상 경계 밖으로 나감

프레임 상태:

- `approved`: 자동 인원수와 모든 자동 박스가 정확함
- `corrected`: 박스를 추가·삭제·수정하거나 인원수를 바로잡음
- `rejected`: 손상·심한 블러·장면 전환 등으로 신뢰할 수 있는 라벨 작성이 불가능함

대부분 `자동 판정`으로 두면 사이트가 변경 여부에 따라 `approved` 또는
`corrected`를 결정합니다. 흑백이거나 기존 모델이 사람을 놓쳤다는 이유로
`rejected`하지 마세요. 직접 박스를 추가한 뒤 `corrected`로 저장해야 모델의
약점을 보완할 수 있습니다.

사람이 없는데 볼라드를 사람으로 인식한 경우:

1. 실제 인원수를 `0`으로 입력합니다.
2. 볼라드 박스를 `삭제(X)` 처리합니다.
3. 장면 상태를 `empty`로 선택합니다.
4. 저장 후 다음으로 이동합니다.

거의 동일한 빈 프레임은 모두 사용할 필요가 없습니다. 같은 장애물·구도라면
10~20장, 조명과 가림 변화까지 포함해 영상당 약 30~50장을 우선 검수합니다.

단축키:

| 키 | 동작 |
|---|---|
| `K` | 선택한 자동 박스 유지 |
| `X` | 선택한 박스 삭제 |
| `R` | 선택한 박스 다시 그리기 |
| `N` | 누락된 사람 박스 추가 |
| `F` | 선택한 사람의 발 위치 지정 |
| `Ctrl+Enter` | 저장 후 다음 프레임 |

## 8. 시계열과 캘리브레이션

- 영상이 편집 없이 이어지면 `이전·다음 프레임과 연속된 장면`을 선택합니다.
- 실제 UTC 촬영 시각을 모르면 비워 둡니다.
- 5분 예측 정답 쌍은 5분 이상의 연속 검수 구간이 있어야 생성됩니다.
- 실제 바닥 가로·세로 길이를 측정하지 않았다면 캘리브레이션을 입력하지 않습니다.
- 임의 면적을 입력하지 않아도 객체 검출·인원수·시계열 데이터는 생성됩니다.

## 9. 학습 데이터 내보내기

화면의 `3. 검증·학습 데이터 생성` 탭에서 다음을 확인합니다.

- 인원수/박스 불일치가 `0`
- 학습 포함 가능 프레임 수가 `1` 이상
- 실제 면적을 모르면 `실측 캘리브레이션 기반 ㎡ 밀도 포함`을 선택하지 않음

그다음 `검증 후 학습 데이터 생성`을 누릅니다. 생성 위치:

```text
data/training/<데이터셋>/exports/export_<생성시각>/
```

주요 결과물:

- `images/train`, `images/val`, `images/test`
- `labels/train`, `labels/val`, `labels/test`
- `data.yaml`
- `quality_report.json`
- 검수된 인원수 및 시계열 CSV

내보낸 폴더를 전달하기 전에 `quality_report.json`을 함께 확인합니다.

## 10. 여러 영상 데이터 합치기

각 영상의 export 폴더가 준비되면 다음처럼 합칩니다. 출력 폴더는 새 이름을
사용해야 합니다.

```bash
python tools/build_combined_yolo_dataset.py \
  --source video3="data/training/sample_data_3/exports/export_생성시각" \
  --source video4="data/training/sample_data_4/exports/export_생성시각" \
  --output "data/training/combined_v3_v4"
```

Windows PowerShell에서는 `\` 대신 한 줄로 실행하거나 백틱을 사용하세요.

## 11. 문제 해결

### `No module named ...`

전용 가상환경을 활성화하고 의존성을 다시 설치합니다.

```bash
python -m pip install -r tools/requirements-training.txt
```

### `Port 8765 is already used`

기존 검수 서버를 `Ctrl+C`로 종료하거나 `-Port 8766`을 사용합니다.

### 이전 영상이 계속 표시됨

서버가 이전 데이터셋으로 실행 중일 수 있습니다. `/api/status`의
`dataset_path`와 `source_video`를 확인하고, 서버를 종료한 뒤 원하는 데이터셋으로
다시 시작합니다. 브라우저는 `Ctrl+F5`로 새로고침합니다.

### 저장 후 다음으로 이동하지 않음

실제 인원수와 활성 박스 수가 다른지 확인합니다. 화면 하단의 저장 실패 메시지에
원인이 표시됩니다.

### 캘리브레이션 면적을 모름

입력하지 않는 것이 맞습니다. 내보내기에서도 실측 밀도 포함을 선택하지 않습니다.

## 12. Git에 올리지 말아야 할 파일

- 원본 영상
- `data/training` 아래의 추출 이미지와 검수 결과
- `.pt`, `.onnx`, `.engine` 모델 파일
- `runs` 학습 결과
- 개인 `.env` 파일

Git에는 `tools` 소스와 문서만 커밋합니다.
