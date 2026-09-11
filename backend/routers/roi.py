from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from services.unavailable_camera_service import InferenceUnavailableError


router = APIRouter(prefix="/api/roi", tags=["roi-density"])


class ROIConfigUpdate(BaseModel):
    camera_id: str = Field(min_length=1, max_length=100)
    roi_points_normalized: list[tuple[float, float]] = Field(
        min_length=4, max_length=4
    )
    zone_width_m: float = Field(gt=0, le=10000)
    zone_height_m: float = Field(gt=0, le=10000)
    grid_rows: int = Field(default=3, ge=1, le=20)
    grid_cols: int = Field(default=3, ge=1, le=20)
    local_window_width_m: float = Field(default=1.0, gt=0, le=10000)
    local_window_height_m: float = Field(default=1.0, gt=0, le=10000)
    local_window_step_m: float = Field(default=0.1, gt=0, le=1000)
    relaxed_max: float = Field(default=2.0, ge=0, le=1000)
    danger_min: float = Field(default=5.0, gt=0, le=1000)


@router.get("")
async def get_roi_config(request: Request):
    camera_id = request.app.state.default_camera_id
    try:
        analyzer = request.app.state.multi_camera_service.get_density_analyzer(camera_id)
        return {"data": analyzer.get_config()}
    except InferenceUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.put("")
async def update_roi_config(payload: ROIConfigUpdate, request: Request):
    return await update_camera_roi_config(payload.camera_id, payload, request)


@router.get("/{camera_id}")
async def get_camera_roi_config(camera_id: str, request: Request):
    try:
        analyzer = request.app.state.multi_camera_service.get_density_analyzer(camera_id)
        return {"data": analyzer.get_config()}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Unknown camera id") from exc
    except InferenceUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.put("/{camera_id}")
async def update_camera_roi_config(
    camera_id: str,
    payload: ROIConfigUpdate,
    request: Request,
):
    try:
        if payload.camera_id != camera_id:
            raise ValueError("Path camera_id and payload camera_id must match")
        analyzer = request.app.state.multi_camera_service.get_density_analyzer(camera_id)
        values = (
            payload.model_dump()
            if hasattr(payload, "model_dump")
            else payload.dict()
        )
        data = analyzer.update_config(values)
        return {"status": "ok", "data": data}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Unknown camera id") from exc
    except InferenceUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
