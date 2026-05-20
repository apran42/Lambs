import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

class Settings:
    # InfluxDB 관련 설정
    INFLUXDB_URL: str = os.getenv("INFLUXDB_URL", "http://localhost:8086")
    INFLUXDB_TOKEN: str = os.getenv("INFLUXDB_TOKEN", "")
    INFLUXDB_ORG: str = os.getenv("INFLUXDB_ORG", "Lambs")
    INFLUXDB_BUCKET: str = os.getenv("INFLUXDB_BUCKET", "crowd_monitor")
    
    # AI & Video 관련 설정
    YOLO_MODEL_PATH: str = os.getenv("YOLO_MODEL_PATH", "../yolov8n.pt")
    VIDEO_PATH: str = os.getenv("VIDEO_PATH", "../data/sample_data (1).mp4") # 경로 변경 예정

settings = Settings()