from datetime import datetime, timezone
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import ASYNCHRONOUS  # 💡 비동기 옵션 임포트
from config import settings

class InfluxDBManager:
    def __init__(self):
        self.client = InfluxDBClient(
            url=settings.INFLUXDB_URL,
            token=settings.INFLUXDB_TOKEN,
            org=settings.INFLUXDB_ORG
        )
        # 💡 [db-influx 파트 반영] 데이터를 백그라운드 버퍼에서 비동기로 밀어 넣도록 설정하여
        # 백엔드 스트리밍 및 AI 워커 루프가 DB I/O 때문에 지연되는 현상을 원천 차단합니다.
        self.write_api = self.client.write_api(write_options=ASYNCHRONOUS)
        
        # 💡 [main 파트 반영] 데이터 조회용 Query API (프론트엔드 통계 API 연동용)
        self.query_api = self.client.query_api()

    def save_crowd_stats(
            self,
            facility_name: str,
            location: str,
            camera_id: str,
            count: int
        ):
        """
        시설물명, 구역 위치, 카메라 ID를 태그로 매핑하여 
        실시간 군중 인원수 데이터를 InfluxDB에 안전하게 적재합니다.
        """
        try:
            point = Point("crowd_stats") \
                .tag("facility", facility_name) \
                .tag("location", location) \
                .tag("device_id", camera_id) \
                .field("people_count", count) \
                .time(datetime.now(timezone.utc))  # 💡 표준 UTC 사용 고정
            
            self.write_api.write(
                bucket=settings.INFLUXDB_BUCKET, 
                org=settings.INFLUXDB_ORG if hasattr(settings, 'INFLUXDB_ORG') else settings.INFLUX_ORG, 
                record=point
            )
        except Exception as e:
            print(f"❌ [DB Error] InfluxDB Write Failed: {e}")

    def get_historical_stats(self, query_string: str):
        """
        Flux 쿼리문을 입력받아 데이터를 조회합니다. (통계 API 연동용)
        """
        try:
            org_name = settings.INFLUXDB_ORG if hasattr(settings, 'INFLUXDB_ORG') else settings.INFLUX_ORG
            return self.query_api.query(org=org_name, query=query_string)
        except Exception as e:
            print(f"❌ [DB Error] InfluxDB Query Failed: {e}")
            return None

    def get_recent_crowd_stats(self, minutes: int = 60):
        """
        get_historical_stats의 아웃풋을 프론트엔드 맞춤형 JSON 포맷으로 가공합니다.
        """
        query = f'''
        from(bucket: "{settings.INFLUXDB_BUCKET}")
          |> range(start: -{minutes}m)
          |> filter(fn: (r) => r["_measurement"] == "crowd_stats")
          |> filter(fn: (r) => r["_field"] == "people_count")
          |> aggregateWindow(every: 1m, fn: mean, createEmpty: false)
          |> yield(name: "mean")
        '''

        # 1. 쿼리 메서드 호출
        tables = self.get_historical_stats(query)
        if not tables:
            return []
            
        # 2. InfluxDB raw 객체를 프론트엔드용 딕셔너리로 가공
        results = []
        for table in tables:
            for record in table.records:
                results.append({
                    "time": record.get_time().isoformat(),
                    "facility": record.values.get("facility"),
                    "location": record.values.get("location"),
                    "camera_id": record.values.get("device_id"),
                    "count": int(record.get_value())
                })
        return results

    def close(self):
        """커넥션 명시적 종료 및 비동기 버퍼 flush"""
        # 💡 [db-influx 파트 반영] 비동기 버퍼의 잔여 데이터를 안전하게 비우고 닫습니다.
        if hasattr(self, 'write_api'):
            self.write_api.close()
        self.client.close()
        print("🔌 InfluxDB 커넥션이 안전하게 종료되었습니다.")

    def __del__(self):
        """인스턴스 소멸 시 안전하게 연결 해제"""
        try:
            self.close()
        except:
            pass

# 싱글톤 인스턴스로 수출
db_manager = InfluxDBManager()