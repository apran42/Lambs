from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

from config import settings


logger = logging.getLogger(__name__)
Timestamp = Optional[Union[datetime, str]]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_timestamp(value: Timestamp) -> datetime:
    if value is None:
        measured_at = _utc_now()
    elif isinstance(value, datetime):
        measured_at = value
    else:
        measured_at = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if measured_at.tzinfo is None:
        measured_at = measured_at.replace(tzinfo=timezone.utc)
    return measured_at.astimezone(timezone.utc)


class InfluxDBManager:
    """Durably queue metrics locally and deliver them to InfluxDB in batches."""

    def __init__(
        self,
        enabled: Optional[bool] = None,
        outbox_path: Optional[str] = None,
        client: Any = None,
        start_worker: bool = True,
    ):
        self.enabled = settings.INFLUXDB_ENABLED if enabled is None else bool(enabled)
        self.outbox_path = Path(outbox_path or settings.INFLUXDB_OUTBOX_PATH)
        self._closed = False
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread = None
        self._connection = None
        self.client = None
        self.write_api = None
        self.query_api = None
        self._available = False
        self._last_success_at = None
        self._last_error = None

        if not self.enabled:
            return

        required_mount = settings.INFLUXDB_OUTBOX_MOUNT
        if required_mount and not os.path.ismount(required_mount):
            self.enabled = False
            self._last_error = "Required outbox mount is not mounted: {}".format(
                required_mount
            )
            logger.error(self._last_error)
            return

        self.outbox_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(
            str(self.outbox_path), check_same_thread=False, timeout=30
        )
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._connection.execute("PRAGMA busy_timeout=30000")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS metrics_outbox (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_key TEXT NOT NULL UNIQUE,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                last_error TEXT
            )
            """
        )
        self._connection.commit()

        self.client = client or InfluxDBClient(
            url=settings.INFLUXDB_URL,
            token=settings.INFLUXDB_TOKEN,
            org=settings.INFLUXDB_ORG,
        )
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        self.query_api = self.client.query_api()
        if start_worker:
            self._thread = threading.Thread(
                target=self._delivery_loop,
                name="influx-outbox",
                daemon=True,
            )
            self._thread.start()

    def save_crowd_stats(
        self,
        facility_name: str,
        location: str,
        camera_id: str,
        count: int,
        timestamp: Timestamp = None,
        roi_count: Optional[int] = None,
        density: Optional[float] = None,
        density_level: Optional[str] = None,
    ) -> bool:
        if self._closed or not self.enabled or self._connection is None:
            return False
        try:
            measured_at = _normalize_timestamp(timestamp)
            payload = {
                "facility": facility_name,
                "location": location,
                "camera_id": camera_id,
                "people_count": int(count),
                "roi_people_count": int(roi_count) if roi_count is not None else None,
                "density_people_per_m2": (
                    float(density) if density is not None else None
                ),
                "density_level": density_level or None,
                "measured_at": measured_at.isoformat(),
            }
            event_key = "{}:{}".format(camera_id, measured_at.isoformat())
            with self._lock:
                self._connection.execute(
                    """
                    INSERT OR IGNORE INTO metrics_outbox
                        (event_key, payload_json, created_at)
                    VALUES (?, ?, ?)
                    """,
                    (
                        event_key,
                        json.dumps(payload, ensure_ascii=False),
                        _utc_now().isoformat(),
                    ),
                )
                self._connection.commit()
            return True
        except Exception as exc:
            self._record_error(exc)
            logger.exception("InfluxDB outbox write failed")
            return False

    def _point_from_payload(self, payload: Dict[str, Any]) -> Point:
        point = (
            Point("crowd_stats")
            .tag("facility", payload["facility"])
            .tag("location", payload["location"])
            .tag("device_id", payload["camera_id"])
            .field("people_count", int(payload["people_count"]))
        )
        if payload.get("roi_people_count") is not None:
            point = point.field("roi_people_count", int(payload["roi_people_count"]))
        if payload.get("density_people_per_m2") is not None:
            point = point.field(
                "density_people_per_m2", float(payload["density_people_per_m2"])
            )
        if payload.get("density_level"):
            point = point.field("density_level", payload["density_level"])
        return point.time(_normalize_timestamp(payload["measured_at"]))

    def _flush_once(self) -> int:
        if not self.enabled or self._connection is None or self.write_api is None:
            return 0
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT id, payload_json
                FROM metrics_outbox
                ORDER BY id
                LIMIT ?
                """,
                (max(1, settings.INFLUXDB_BATCH_SIZE),),
            ).fetchall()
        if not rows:
            self._refresh_health()
            return 0

        row_ids = [row[0] for row in rows]
        try:
            points = [self._point_from_payload(json.loads(row[1])) for row in rows]
            self.write_api.write(
                bucket=settings.INFLUXDB_BUCKET,
                org=settings.INFLUXDB_ORG,
                record=points,
            )
            with self._lock:
                placeholders = ",".join("?" for _ in row_ids)
                self._connection.execute(
                    "DELETE FROM metrics_outbox WHERE id IN ({})".format(placeholders),
                    row_ids,
                )
                self._connection.commit()
            self._available = True
            self._last_success_at = _utc_now().isoformat()
            self._last_error = None
            return len(rows)
        except Exception as exc:
            with self._lock:
                placeholders = ",".join("?" for _ in row_ids)
                self._connection.execute(
                    """
                    UPDATE metrics_outbox
                    SET attempts = attempts + 1, last_error = ?
                    WHERE id IN ({})
                    """.format(placeholders),
                    [str(exc)] + row_ids,
                )
                self._connection.commit()
            self._record_error(exc)
            logger.warning("InfluxDB delivery failed; metrics remain queued: %s", exc)
            return 0

    def _refresh_health(self) -> None:
        try:
            health = self.client.health()
            self._available = getattr(health, "status", "") == "pass"
            if self._available:
                self._last_error = None
            else:
                self._last_error = getattr(health, "message", "InfluxDB health failed")
        except Exception as exc:
            self._record_error(exc)

    def _record_error(self, exc: Exception) -> None:
        self._available = False
        self._last_error = str(exc)

    def _delivery_loop(self) -> None:
        while not self._stop_event.is_set():
            self._flush_once()
            self._stop_event.wait(max(0.1, settings.INFLUXDB_RETRY_SECONDS))

    def health(self) -> Dict[str, Any]:
        pending = 0
        if self._connection is not None:
            try:
                with self._lock:
                    pending = int(
                        self._connection.execute(
                            "SELECT COUNT(*) FROM metrics_outbox"
                        ).fetchone()[0]
                    )
            except Exception as exc:
                self._record_error(exc)
        return {
            "enabled": self.enabled,
            "available": self._available,
            "url": settings.INFLUXDB_URL,
            "bucket": settings.INFLUXDB_BUCKET,
            "outbox_path": str(self.outbox_path) if self.enabled else None,
            "pending_rows": pending,
            "last_success_at": self._last_success_at,
            "last_error": self._last_error,
        }

    def get_recent_crowd_stats(self, minutes: int = 60):
        if not self.enabled or self.query_api is None:
            raise RuntimeError("InfluxDB storage is disabled")
        query = '''from(bucket: "{}")
          |> range(start: -{}m)
          |> filter(fn: (r) => r["_measurement"] == "crowd_stats")
          |> filter(fn: (r) => r["_field"] == "people_count")
          |> aggregateWindow(every: 1m, fn: mean, createEmpty: false)
          |> yield(name: "mean")'''.format(settings.INFLUXDB_BUCKET, minutes)
        try:
            tables = self.query_api.query(org=settings.INFLUXDB_ORG, query=query)
        except Exception as exc:
            self._record_error(exc)
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

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, settings.INFLUXDB_RETRY_SECONDS + 1.0))
        if self.enabled and self._connection is not None:
            self._flush_once()
        if self.write_api is not None:
            self.write_api.close()
        if self.client is not None:
            self.client.close()
        if self._connection is not None:
            self._connection.close()


db_manager = InfluxDBManager()
