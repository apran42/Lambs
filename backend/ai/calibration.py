import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


CHESSBOARD_SIZE = (9, 6)
SQUARE_SIZE_MM = 25.0


def calibrate_camera(image_dir: str, output_path: str):
    directory = Path(image_dir)
    image_paths = sorted((*directory.glob("*.jpg"), *directory.glob("*.png")))
    if not image_paths:
        return None

    object_template = np.zeros(
        (CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32
    )
    object_template[:, :2] = np.mgrid[
        0 : CHESSBOARD_SIZE[0], 0 : CHESSBOARD_SIZE[1]
    ].T.reshape(-1, 2)
    object_template *= SQUARE_SIZE_MM

    object_points = []
    image_points = []
    image_size = None
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )

    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        current_size = gray.shape[::-1]
        if image_size is None:
            image_size = current_size
        elif current_size != image_size:
            raise ValueError("All calibration images must have the same resolution")
        found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE)
        if found:
            image_points.append(
                cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            )
            object_points.append(object_template.copy())

    if len(object_points) < 5 or image_size is None:
        return None

    rms, camera_matrix, distortion, rotations, translations = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    if not np.isfinite(rms):
        raise RuntimeError("Camera calibration did not converge")

    errors = []
    for obj, img, rotation, translation in zip(
        object_points, image_points, rotations, translations
    ):
        projected, _ = cv2.projectPoints(
            obj, rotation, translation, camera_matrix, distortion
        )
        errors.append(cv2.norm(img, projected, cv2.NORM_L2) / len(projected))

    data = {
        "version": datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
        "image_size": list(image_size),
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": distortion.tolist(),
        "rms_error": float(rms),
        "reprojection_error": float(np.mean(errors)),
        "sample_count": len(object_points),
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="./calib_images")
    parser.add_argument("--output", default="./calibration_data.json")
    arguments = parser.parse_args()
    calibrate_camera(arguments.images, arguments.output)
