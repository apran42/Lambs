from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone

from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import ASYNCHRONOUS

from config import settings


logger = logging.getLogger(__name__)


class InfluxDBManager:
    def __init__(self):
        self.client = InfluxDBClient(
            url=settings.INFLUXDB_URL,
            token=settings.INFLUXDB_TOKEN,
            org=settings.INFLUXDB_ORG,
        )
        self.write_api = self.client.write_api(write_options=ASYNCHRONOUS)
        self.query_api = self.client.query_api()
        self._closed = False
        self._close_lock = threading.Lock()

    def save_crowd_stats(
        self,
        facility_name: str,
        location: str,
        camera_id: str,
        count: int,
        timestamp: datetime | None = None,
        roi_count: int | None = None,
        density: float | None = None,
        density_level: str | None = None,
    ) -> bool:
        if self._closed:
            return False
        measured_at = timestamp or datetime.now(timezone.utc)
        if measured_at.tzinfo is None:
            measured_at = measured_at.replace(tzinfo=timezone.utc)
        try:
            point = (
                Point("crowd_stats")
                .tag("facility", facility_name)
                .tag("location", location)
                .tag("device_id", camera_id)
                .field("people_count", int(count))
            )
            if roi_count is not None:
                point = point.field("roi_people_count", int(roi_count))
            if density is not None:
                point = point.field("density_people_per_m2", float(density))
            if density_level:
                point = point.field("density_level", density_level)
            point = point.time(measured_at)
            self.write_api.write(
                bucket=settings.INFLUXDB_BUCKET,
                org=settings.INFLUXDB_ORG,
                record=point,
            )
            return True
        except Exception:
            logger.exception("InfluxDB write failed")
            return False

    def get_recent_crowd_stats(self, minutes: int = 60):
        query = f'''from(bucket: "{settings.INFLUXDB_BUCKET}")
          |> range(start: -{minutes}m)
          |> filter(fn: (r) => r["_measurement"] == "crowd_stats")
          |> filter(fn: (r) => r["_field"] == "people_count")
          |> aggregateWindow(every: 1m, fn: mean, createEmpty: false)
          |> yield(name: "mean")'''
        try:
            tables = self.query_api.query(org=settings.INFLUXDB_ORG, query=query)
        except Exception:
            logger.exception("InfluxDB query failed")
            raise
        return [
            {
                "time": record.get_time().isoformat(),
                "facility": record.values.get("facility"),
                "location": record.values.get("location"),
                "camera_id": record.values.get("device_id"),
                "count": float(record.get_value()),
            }
            for table in tables
            for record in table.records
        ]

    def close(self):
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            self.write_api.close()
            self.client.close()


db_manager = InfluxDBManager()
