import asyncio
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from database.influx_client import db_manager


router = APIRouter(prefix="/api/metrics", tags=["metrics"])


class MetricEntry(BaseModel):
    camera_id: str = Field(min_length=1, max_length=100)
    facility: str = Field(min_length=1, max_length=100)
    location: str = Field(min_length=1, max_length=100)
    timestamp: datetime | None = None
    count: int = Field(ge=0)


@router.post("/bulk")
async def write_metrics_bulk(entries: list[MetricEntry]):
    if len(entries) > 1000:
        raise HTTPException(status_code=413, detail="A batch may contain at most 1000 rows")
    failed = 0
    for entry in entries:
        written = db_manager.save_crowd_stats(
            facility_name=entry.facility,
            location=entry.location,
            camera_id=entry.camera_id,
            count=entry.count,
            timestamp=entry.timestamp,
        )
        failed += not written
    if failed:
        raise HTTPException(status_code=503, detail=f"Failed to queue {failed} rows")
    return {"status": "ok", "written": len(entries)}


@router.get("/recent")
async def get_recent_metrics(
    minutes: Annotated[int, Query(ge=1, le=1440)] = 5,
):
    try:
        data = await asyncio.to_thread(db_manager.get_recent_crowd_stats, minutes)
        return {"data": data, "minutes": minutes}
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Metrics storage is unavailable") from exc
