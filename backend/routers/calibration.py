import asyncio
from functools import partial

from fastapi import APIRouter, HTTPException

from config import settings


router = APIRouter(prefix="/api/calibration", tags=["calibration"])


@router.post("/run")
async def run_calibration():
    try:
        # Keep OpenCV out of the API-only import path. Calibration is available
        # only when its optional runtime dependency is installed.
        from ai.calibration import calibrate_camera

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            partial(
                calibrate_camera,
                settings.CALIBRATION_IMAGES_PATH,
                settings.CALIBRATION_PATH,
            ),
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
    except ModuleNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="Calibration runtime is not installed in API-only mode.",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
