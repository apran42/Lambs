from fastapi import APIRouter, Query

from routers.metrics import get_recent_metrics


router = APIRouter(prefix="/api/stats", tags=["stats"])


@router.get("/recent", deprecated=True)
async def get_recent_stats(
    minutes: int = Query(60, ge=1, le=1440),
):
    """Compatibility endpoint. Prefer /api/metrics/recent."""
    return await get_recent_metrics(minutes)
