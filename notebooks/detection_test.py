"""바벨 검출 테스트 v2 - YOLOv8-Pose로 관절 키포인트 추출"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.video import load_video, save_frame
from ultralytics import YOLO
import cv2

frames, fps, size = load_video("data/videos/bench.mp4")

# YOLOv8-Pose 모델 (관절 17개 추출)
model = YOLO("yolov8n-pose.pt")

# 키포인트 인덱스 (COCO 기준)
# 5: 왼쪽 어깨, 6: 오른쪽 어깨
# 7: 왼쪽 팔꿈치, 8: 오른쪽 팔꿈치
# 9: 왼쪽 손목, 10: 오른쪽 손목

KEYPOINT_NAMES = {
    5: "L_shoulder", 6: "R_shoulder",
    7: "L_elbow", 8: "R_elbow",
    9: "L_wrist", 10: "R_wrist",
}

sample_indices = [30, 90, 150, 200, 250]
sample_dir = Path("data/outputs/samples")
sample_dir.mkdir(parents=True, exist_ok=True)

for idx in sample_indices:
    results = model(frames[idx], conf=0.3)
    print(f"\n=== 프레임 {idx} ({idx/fps:.1f}초) ===")

    for r in results:
        if r.keypoints is not None and len(r.keypoints) > 0:
            kpts = r.keypoints[0].data[0]  # 첫 번째 사람의 키포인트

            for kpt_idx, name in KEYPOINT_NAMES.items():
                x, y, conf = kpts[kpt_idx]
                if conf > 0.3:
                    print(f"  {name}: ({x:.0f}, {y:.0f}) conf={conf:.2f}")
                else:
                    print(f"  {name}: 검출 실패")

            # 손목 좌표로 바벨 기울기 계산
            lw_x, lw_y, lw_c = kpts[9]
            rw_x, rw_y, rw_c = kpts[10]
            if lw_c > 0.3 and rw_c > 0.3:
                import math
                angle = math.degrees(math.atan2(abs(rw_y - lw_y), abs(rw_x - lw_x)))
                print(f"  >> 바벨 추정 기울기: {angle:.1f}도")
        else:
            print("  관절 검출 실패")

    # 결과 이미지 저장
    annotated = results[0].plot()
    save_frame(annotated, sample_dir / f"frame_{idx}_pose.jpg")

print("\n완료! data/outputs/samples/ 에서 _pose.jpg 확인해봐")
