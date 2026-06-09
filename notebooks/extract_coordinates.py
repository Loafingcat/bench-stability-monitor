"""전체 프레임 좌표 추출"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.video import load_video
from src.detection.plate_tracker import track_all_frames, save_tracking_csv

frames, fps, size = load_video("data/videos/bench.mp4")

print("좌표 추출 시작...")
df = track_all_frames(frames, fps)

save_tracking_csv(df, "data/outputs/bench_tracking.csv")

# 간단한 요약
print(f"\n총 프레임: {len(df)}")
print(f"손목 검출 성공률: {df['L_wrist_conf'].gt(0.3).mean():.1%} (왼쪽), {df['R_wrist_conf'].gt(0.3).mean():.1%} (오른쪽)")
print(f"바벨 기울기 범위: {df['bar_angle'].min():.1f}도 ~ {df['bar_angle'].max():.1f}도")
