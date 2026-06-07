"""
카메라 왜곡 보정 파라미터를 체스보드 이미지로부터 계산하고 JSON으로 저장합니다.

사용법:
    python -m ai.calibration --images ./calib_images --output ./calibration_data.json

calib_images/ 폴더에 체스보드 사진(jpg/png)을 넣고 실행하면
camera_matrix, dist_coeffs, 버전 정보가 JSON으로 저장됩니다.
"""


import cv2
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime, timezone


# 체스보드 내부 코너 수 (가로, 세로) — 실제 보드에 맞게 수정
CHESSBOARD_SIZE = (9, 6)
# 체스보드 한 칸의 실제 크기 (mm 단위)
SQUARE_SIZE_MM = 25.0


def calibrate_camera(image_dir: str, output_path: str):
    """
    이미지 폴더에서 체스보드 코너를 검출하고
    camera_matrix / dist_coeffs 를 계산하여 JSON으로 저장합니다.
    """
    image_dir = Path(image_dir)
    image_paths = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))

    if not image_paths:
        print(f"❌ 이미지를 찾을 수 없습니다: {image_dir}")
        return None

    # 3D 실세계 좌표 기준점 생성
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM

    obj_points = []  # 3D 실세계 좌표
    img_points = []  # 2D 이미지 좌표

    image_size = None

    for path in image_paths:
        img = cv2.imread(str(path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]

        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
        if ret:
            # 서브픽셀 정밀도로 코너 위치 개선
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_points.append(objp)
            img_points.append(corners_refined)
            print(f"✅ 코너 검출 성공: {path.name}")
        else:
            print(f"⚠️ 코너 검출 실패: {path.name}")

    if len(obj_points) < 5:
        print(f"❌ 유효한 이미지가 5장 미만입니다. ({len(obj_points)}장 검출됨)")
        return None

    # 캘리브레이션 계산
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )

    # 재투영 오차 계산 (낮을수록 정확, 1.0 이하 권장)
    mean_error = 0.0
    for i in range(len(obj_points)):
        projected, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        mean_error += cv2.norm(img_points[i], projected, cv2.NORM_L2) / len(projected)
    reprojection_error = mean_error / len(obj_points)

    print(f"📐 재투영 오차: {reprojection_error:.4f} (1.0 이하 권장)")

    # JSON 저장
    calibration_data = {
        "version": datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
        "image_size": list(image_size),
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.tolist(),
        "reprojection_error": reprojection_error,
        "sample_count": len(obj_points)
    }

    with open(output_path, "w") as f:
        json.dump(calibration_data, f, indent=2)

    print(f"💾 캘리브레이션 데이터 저장 완료: {output_path}")
    return calibration_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="카메라 캘리브레이션 스크립트")
    parser.add_argument("--images", type=str, default="./calib_images", help="체스보드 이미지 폴더 경로")
    parser.add_argument("--output", type=str, default="./calibration_data.json", help="출력 JSON 경로")
    args = parser.parse_args()

    calibrate_camera(args.images, args.output)
