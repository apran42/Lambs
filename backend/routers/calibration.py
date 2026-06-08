# routers/calibration.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
# 💡 핵심: ai 패키지에 정의된 calibrate_camera 함수를 가져옵니다.
from ai.calibration import calibrate_camera  

# 💡 main.py가 간절히 찾고 있던 바로 그 'router' 변수입니다!
router = APIRouter(prefix="/api/calibration", tags=["calibration"])

# API 요청 시 프론트엔드로부터 받을 데이터 규격 정의
class CalibRequest(BaseModel):
    image_dir: str = "./calib_images"
    output_path: str = "./calibration_data.json"

@router.post("/run")
async def run_calibration(req: CalibRequest):
    """
    웹 API 요청(POST)을 받아 카메라 왜곡 보정(Calibration)을 실행합니다.
    """
    try:
        # ai/calibration.py 파일 안에 있는 실제 연산 함수를 호출합니다.
        result = calibrate_camera(req.image_dir, req.output_path)
        
        if result is None:
            raise HTTPException(
                status_code=400, 
                detail="캘리브레이션 실패. 이미지 경로에 체스보드 사진이 충분한지 확인하세요."
            )
            
        return {
            "status": "ok", 
            "message": "카메라 왜곡 보정이 성공적으로 완료되었습니다.", 
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))