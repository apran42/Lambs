import cv2
import numpy as np


def build_homography(src_points: list, dst_points: list):
    """
    4개의 대응점으로 호모그래피 행렬을 계산합니다.

    [Input]
      src_points: 원본 프레임의 4개 픽셀 좌표 [[x,y], ...]
                  (카메라 시점 기준 — 좌상, 우상, 우하, 좌하 순서 권장)
      dst_points: bird-eye view에서의 4개 대응 좌표 [[x,y], ...]
                  (실세계 평면 좌표 또는 정규화 좌표)

    [Output] 3x3 호모그래피 행렬 (numpy array)
    """
    src = np.array(src_points, dtype=np.float32)
    dst = np.array(dst_points, dtype=np.float32)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC)
    return H


def to_bird_eye(pixel_x: float, pixel_y: float, H: np.ndarray):
    """
    단일 픽셀 좌표를 호모그래피 행렬로 bird-eye view 좌표로 변환합니다.

    [Input]
      pixel_x, pixel_y: 원본 프레임 픽셀 좌표
      H: build_homography()로 계산된 3x3 행렬

    [Output] (bx, by) bird-eye view 좌표 (float tuple)
    """
    pt = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
    result = cv2.perspectiveTransform(pt, H)
    bx, by = result[0][0]
    return float(bx), float(by)


def calculate_positions(results, H: np.ndarray = None):
    """
    YOLO의 추론 결과 오브젝트(results)를 받아 바운딩 박스 좌표를 추출하고,
    CCTV 왜곡 보정이나 실제 구역 내 위치 좌표 연산을 수행합니다.

    [Input]
      results: Ultralytics YOLO의 results 객체
      H: (선택) build_homography()로 계산된 호모그래피 행렬
         전달 시 각 박스의 풋포인트를 bird-eye 좌표로 변환하여 함께 반환

    [Output]
      H 없음: [[x1, y1, x2, y2], ...]
      H 있음: [[x1, y1, x2, y2, bx, by], ...]  (bx, by = bird-eye 좌표)
    """
    # 뼈대 코드: 결과에서 바운딩 박스 좌표 리스트 추출
    if results and results[0].boxes:
        boxes = results[0].boxes.xyxy.tolist()
        cls_list = results[0].boxes.cls.tolist()
        orig_h, orig_w = results[0].orig_shape
    else:
        return []

    # 💡 [팀원 2의 알고리즘 작성 공간]
    # 캔버스 고정 해상도
    FRAME_W, FRAME_H = orig_w, orig_h

    MIN_BOX_W, MIN_BOX_H = FRAME_W * 0.01, FRAME_H * 0.01
    MAX_BOX_W, MAX_BOX_H = FRAME_W * 0.95, FRAME_H * 0.95

    processed_boxes = []
    for box, cls_id in zip(boxes, cls_list):
        # 1. 사람(class 0)만 통과
        if int(cls_id) != 0:
            continue

        x1, y1, x2, y2 = box

        # 2. 좌표 클리핑 (화면 밖 박스 제거)
        x1 = max(0.0, min(x1, FRAME_W))
        y1 = max(0.0, min(y1, FRAME_H))
        x2 = max(0.0, min(x2, FRAME_W))
        y2 = max(0.0, min(y2, FRAME_H))

        # 3. 박스 크기 필터링 (노이즈/오탐 제거)
        w, h = x2 - x1, y2 - y1
        if w < MIN_BOX_W or h < MIN_BOX_H:
            continue
        if w > MAX_BOX_W or h > MAX_BOX_H:
            continue

        # 4. 호모그래피가 있으면 풋포인트(발 위치)를 bird-eye 좌표로 변환
        if H is not None:
            foot_x = x1 + w / 2  # 박스 하단 중앙
            foot_y = y2
            bx, by = to_bird_eye(foot_x, foot_y, H)
            processed_boxes.append([x1, y1, x2, y2, bx, by])
        else:
            processed_boxes.append([x1, y1, x2, y2])

    return processed_boxes
