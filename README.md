# Lambs
CCTV 실시간 혼잡도 분석 및 예측 시스템

CCTV 영상을 활용한 실시간 인파 밀집도 분석 및 시계열 예측 졸업작품입니다.

## 🛠 기술 스택
- **AI**: YOLOv8
- **Backend**: FastAPI, InfluxDB
- **Frontend**: React, Konva.js
- **Hardware**: Jetson Nano (Target)

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
### python(3.9+)
```bash
cd backend
pip install -r requirements.txt
```
### 서버 시작
```bash
cd backend
uvicorn main:app --reload
```

## 프론트엔드
### 서버 시작
```bash
cd frontend
npm run dev
```