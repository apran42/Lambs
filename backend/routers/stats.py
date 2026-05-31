from fastapi import APIRouter
from database.influx_client import db_manager

router = APIRouter(prefix="/api/stats", tags=["stats"])

@router.get("/recent")
async def get_recent_stats(minutes: int = 60):
    """
    최근 N분간의 혼잡도 데이터를 반환합니다.
    - 기본값: 60분
    - 프론트엔드 시간대별 그래프용
    """
    data = db_manager.get_recent_crowd_stats(minutes)
    return {"data": data}
