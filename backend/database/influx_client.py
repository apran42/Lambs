from datetime import datetime, timezone
from influxdb_client import InfluxDBClient, Point
from config import settings

class InfluxDBManager:
    def __init__(self):
        self.client = InfluxDBClient(
            url=settings.INFLUXDB_URL,
            token=settings.INFLUXDB_TOKEN,
            org=settings.INFLUXDB_ORG
        )
        # 데이터를 빠르게 밀어 넣기 위한 Write API (동기 방식)
        self.write_api = self.client.write_api()

    def save_crowd_stats(self, camera_id: str, location: str, count: int):
        try:
            point = Point("crowd_stats") \
                .tag("device_id", camera_id) \
                .tag("location", location) \
                .field("people_count", count) \
                .time(datetime.now(timezone.utc))  # Deprecated된 utcnow() 대신 표준 UTC 사용
            
            self.write_api.write(
                bucket=settings.INFLUXDB_BUCKET, 
                org=settings.INFLUXDB_ORG, 
                record=point
            )
        except Exception as e:
            print(f"❌ [DB Error] InfluxDB Write Failed: {e}")

    def close(self):
        self.client.close()

# 싱글톤 인스턴스로 수출하여 어디서든 하나의 커넥션으로 공유
db_manager = InfluxDBManager()