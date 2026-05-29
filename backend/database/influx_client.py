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
        # 데이터 적재용 Write API (기본 배치 모드로 고성능 처리)
        self.write_api = self.client.write_api()
        # 데이터 조회용 Query API (추후 통계 API 구현용)
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
                .time(datetime.now(timezone.utc))
            
            self.write_api.write(
                bucket=settings.INFLUXDB_BUCKET, 
                org=settings.INFLUXDB_ORG if hasattr(settings, 'INFLUXDB_ORG') else settings.INFLUX_ORG, 
                record=point
            )
        except Exception as e:
            print(f"❌ [DB Error] InfluxDB Write Failed: {e}")

    def get_historical_stats(self, query_string: str):
        """
        Flux 쿼리문을 입력받아 데이터를 조회합니다. (추후 API 연동용)
        """
        try:
            org_name = settings.INFLUXDB_ORG if hasattr(settings, 'INFLUXDB_ORG') else settings.INFLUX_ORG
            return self.query_api.query(org=org_name, query=query_string)
        except Exception as e:
            print(f"❌ [DB Error] InfluxDB Query Failed: {e}")
            return None

    def get_recent_crowd_stats(self, minutes: int = 60):
        """
        [새로 추가] get_historical_stats의 아웃풋을 프론트엔드 맞춤형 JSON 포맷으로 가공합니다.
        """
        query = f'''
        from(bucket: "{settings.INFLUXDB_BUCKET}")
          |> range(start: -{minutes}m)
          |> filter(fn: (r) => r["_measurement"] == "crowd_stats")
          |> filter(fn: (r) => r["_field"] == "people_count")
          |> aggregateWindow(every: 1m, fn: mean, createEmpty: false)
          |> yield(name: "mean")
        '''

        # 1. 내가 만든 기존 쿼리 메서드 호출
        tables = self.get_historical_stats(query)
        if not tables:
            return []
            
        # 2. 복잡한 InfluxDB raw 객체를 프론트엔드용 딕셔너리로 세련되게 가공
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
        """커넥션 명시적 종료"""
        self.client.close()

    def __del__(self):
        """인스턴스 소멸 시 안전하게 연결 해제"""
        try:
            self.close()
        except:
            pass

# 싱글톤 인스턴스로 수출
db_manager = InfluxDBManager()