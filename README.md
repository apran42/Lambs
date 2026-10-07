# Lambs
CCTV 실시간 혼잡도 분석 및 예측 시스템

CCTV 영상을 활용한 실시간 인파 밀집도 분석 및 시계열 예측 졸업작품입니다.

학습 영상 프레임 추출, 인원수·바운딩박스 통합 검수, YOLO 데이터 생성 방법은
[`docs/training-data-review-guide.md`](docs/training-data-review-guide.md)를 참고하세요.

## 🛠 기술 스택
- **인공지능**: YOLOv8
- **백엔드**: FastAPI, InfluxDB
- **프론트엔드**: React, Konva.js
- **하드웨어**: Jetson Nano 4GB(시연용 TensorRT 워커·FastAPI 배포)

## 👥 팀원 역할
- **프론트엔드**: 실시간 대시보드 및 Konva 가시화
- **백엔드/AI A**: 영상 처리 파이프라인 및 YOLO 연동
- **백엔드/AI B**: InfluxDB 설계 및 API 개발
- **백엔드/AI C**: 모델 최적화(TensorRT) 및 배포 환경 구축

## 📁 프로젝트 구조
```
lambs/
├── backend/
│   ├── main.py                      # FastAPI 메인 서버 (WebSocket 스트리밍)
│   ├── config.py                    # 환경설정 (InfluxDB, YOLOv8 경로)
│   ├── requirements.txt             # Python 의존성
│   ├── .env.example                 # 환경변수 템플릿
│   ├── ai/
│   │   ├── detector.py              # YOLOv8 객체 탐지 및 추적
│   │   ├── camera.py                # 영상 스트림 처리 (비디오 파일)
│   │   └── __init__.py
│   ├── database/
│   │   ├── influx_client.py         # InfluxDB 연결 및 데이터 저장
│   │   └── __init__.py
│   ├── utils/
│   │   ├── geometry.py              # 바운딩박스 좌표 처리
│   │   └── __init__.py
│   └── .env.example
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx                  # 실시간 대시보드 메인 컴포넌트
│   │   ├── main.jsx                 # React 진입점
│   │   ├── index.css                # 전역 스타일
│   │   ├── App.css                  # App 스타일
│   │   └── assets/                  # 이미지 자산
│   ├── public/                      # 정적 자산 (favicon, icons)
│   ├── package.json                 # npm 의존성
│   ├── vite.config.js               # Vite 설정
│   ├── eslint.config.js             # ESLint 설정
│   └── index.html                   # HTML 진입점
│
└── README.md
```
## 백엔드
### Python (3.10+)
```bash
cd backend
pip install -r requirements.txt
```
### 서버 시작
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

## 프론트엔드
### 서버 시작
```bash
cd frontend
npm run dev
```

현재 젯슨 시연은 영상 파일 1개를 TensorRT 워커(Python 3.6)에서 분석하고,
FastAPI(Python 3.8)가 WebSocket으로 React 관제 화면에 전달합니다. InfluxDB는
외부 장치에서 실행할 수 있으며 젯슨은 적재 실패에 대비한 로컬 Outbox를 사용합니다.
카메라 확장 구조는 있지만 2대 동시 20 FPS와 실제 면적 기준 밀도는 아직 검증되지
않았습니다. 1분 뒤 인원 예측은 현재값 유지 기준선이 기본이고 GRU는 선택적 실험
모드입니다. 실행·설정은 [`backend/README.md`](backend/README.md), 데이터 검수는
[`docs/training-data-review-guide.md`](docs/training-data-review-guide.md)를 참고하세요.
이전 개발 계획은 [`docs/implementation-roadmap.md`](docs/implementation-roadmap.md)에 있습니다.
