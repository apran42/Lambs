import tempfile
from pathlib import Path

from database.influx_client import InfluxDBManager


class FakeHealth:
    status = "pass"
    message = "ready"


class FakeWriteApi:
    def __init__(self):
        self.records = []
        self.error = None

    def write(self, **kwargs):
        if self.error is not None:
            raise self.error
        self.records.extend(kwargs["record"])

    def close(self):
        return None


class FakeQueryApi:
    def query(self, **_kwargs):
        return []


class FakeInfluxClient:
    def __init__(self):
        self.writer = FakeWriteApi()

    def write_api(self, **_kwargs):
        return self.writer

    def query_api(self):
        return FakeQueryApi()

    def health(self):
        return FakeHealth()

    def close(self):
        return None


def test_outbox_retries_and_deduplicates_metrics():
    with tempfile.TemporaryDirectory() as directory:
        fake_client = FakeInfluxClient()
        manager = InfluxDBManager(
            enabled=True,
            outbox_path=str(Path(directory) / "outbox.sqlite3"),
            client=fake_client,
            start_worker=False,
        )
        timestamp = "2026-09-18T10:00:00+00:00"
        arguments = {
            "facility_name": "model-zone",
            "location": "zone-a",
            "camera_id": "cam-01",
            "count": 4,
            "timestamp": timestamp,
            "roi_count": 3,
            "density": 3.0,
            "density_level": "caution",
        }

        assert manager.save_crowd_stats(**arguments)
        assert manager.save_crowd_stats(**arguments)
        assert manager.health()["pending_rows"] == 1

        fake_client.writer.error = RuntimeError("offline")
        assert manager._flush_once() == 0
        failed_health = manager.health()
        assert failed_health["pending_rows"] == 1
        assert failed_health["available"] is False
        assert "offline" in failed_health["last_error"]

        fake_client.writer.error = None
        assert manager._flush_once() == 1
        healthy = manager.health()
        assert healthy["pending_rows"] == 0
        assert healthy["available"] is True
        assert healthy["last_success_at"] is not None
        assert len(fake_client.writer.records) == 1
        manager.close()
