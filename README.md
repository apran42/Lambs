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

## 백엔드 필요한 패키지
## python(3.9+)
```bash
cd backend
pip install -r requirements.txt