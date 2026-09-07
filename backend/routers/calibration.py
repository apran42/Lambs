import asyncio

from fastapi import APIRouter, HTTPException

from ai.calibration import calibrate_camera
from config import settings


router = APIRouter(prefix="/api/calibration", tags=["calibration"])


@router.post("/run")
async def run_calibration():
    try:
        result = await asyncio.to_thread(
            calibrate_camera,
            settings.CALIBRATION_IMAGES_PATH,
            settings.CALIBRATION_PATH,
        )
        if result is None:
            raise HTTPException(
                status_code=400,
                detail="At least five valid chessboard images are required.",
            )
        return {
            "status": "ok",
            "restart_required": True,
            "data": result,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
