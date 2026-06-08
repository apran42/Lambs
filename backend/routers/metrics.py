from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from database.influx_client import db_manager

router = APIRouter(prefix="/api/metrics", tags=["metrics"])

class MetricEntry(BaseModel):
    camera_id: str
    facility: str
    location: str
    zone_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    count: int
    avg_bbox_area: Optional[float] = None
    avg_confidence: Optional[float] = None

@router.post("/bulk")
async def write_metrics_bulk(entries: List[MetricEntry]):
    """
    다중 카메라 메트릭을 Influx에 동시 적재
    """
    try:
        for e in entries:
            db_manager.save_crowd_stats(
                facility_name=e.facility,
                location=e.location,
                camera_id=e.camera_id,
                count=e.count,
            )
        return {"status": "ok", "written": len(entries)}
    except Exception as ex:
        print(f"❌ [API Error] POST /metrics/bulk failed: {ex}")
        raise HTTPException(status_code=500, detail=str(ex))

@router.get("/recent")
async def get_recent_metrics(minutes: int = 5):
    """
    최근 N분간의 혼잡도 데이터 조회
    """
    try:
        data = db_manager.get_recent_crowd_stats(minutes)
        return {"data": data, "minutes": minutes}
    except Exception as ex:
        print(f"❌ [API Error] GET /metrics/recent failed: {ex}")
        raise HTTPException(status_code=500, detail=str(ex))