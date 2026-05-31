def calculate_positions(results):
    """
    YOLO의 추론 결과 오브젝트(results)를 받아 바운딩 박스 좌표를 추출하고,
    CCTV 왜곡 보정이나 실제 구역 내 위치 좌표 연산을 수행합니다.
    
    [Input]  Ultralytics YOLO의 results 객체
    [Output] 가공 또는 필터링이 완료된 최종 bounding boxes 리스트 [[x1, y1, x2, y2], ...]
    """
    # 뼈대 코드: 결과에서 바운딩 박스 좌표 리스트 추출
    if results and results[0].boxes:
        boxes = results[0].boxes.xyxy.tolist()
        cls_list = results[0].boxes.cls.tolist()  # ← 클래스 ID 추가 추출
    else:
        boxes = []
        cls_list = []
        
    # 💡 [팀원 2의 알고리즘 작성 공간]
    # 캔버스 고정 해상도
    FRAME_W, FRAME_H = 640, 480
    MIN_BOX_W, MIN_BOX_H = 15, 30
    MAX_BOX_W, MAX_BOX_H = FRAME_W * 0.8, FRAME_H * 0.8

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
        processed_boxes.append([x1, y1, x2, y2])
    
    return processed_boxes
