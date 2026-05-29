from datetime import datetime, timezone
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import ASYNCHRONOUS # 💡 비동기 옵션 임포트
from config import settings

class InfluxDBManager:
    def __init__(self):
        self.client = InfluxDBClient(
            url=settings.INFLUXDB_URL,
            token=settings.INFLUXDB_TOKEN,
            org=settings.INFLUXDB_ORG
        )
        # 💡 [최적화] 데이터를 백그라운드 버퍼에서 비동기로 밀어 넣도록 설정하여
        # 백엔드 스트리밍 및 AI 워커 루프가 DB I/O 때문에 지연되는 현상을 원천 차단합니다.
        self.write_api = self.client.write_api(write_options=ASYNCHRONOUS)

    def save_crowd_stats(
            self,
            facility_name: str,
            location: str,
            camera_id: str,
            count: int
        ):
        try:
            point = Point("crowd_stats") \
                .tag("facility", facility_name)    \
                .tag("device_id", camera_id) \
                .tag("location", location) \
                .field("people_count", count) \
                .time(datetime.now(timezone.utc))  # 표준 UTC 사용
            
            self.write_api.write(
                bucket=settings.INFLUXDB_BUCKET, 
                org=settings.INFLUXDB_ORG, 
                record=point
            )
        except Exception as e:
            print(f"❌ [DB Error] InfluxDB Write Failed: {e}")

    def close(self):
        # 💡 비동기 버퍼에 남아있는 데이터를 안전하게 비우고(flush) 연결을 닫습니다.
        if hasattr(self, 'write_api'):
            self.write_api.close()
        self.client.close()
        print("🔌 InfluxDB 커넥션이 안전하게 종료되었습니다.")

# 싱글톤 인스턴스로 수출하여 어디서든 하나의 커넥션으로 공유
db_manager = InfluxDBManager()