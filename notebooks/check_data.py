"""추출 데이터 확인 v2"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

df = pd.read_csv("data/outputs/bench_tracking.csv")

print("=== 손목 높이 차이 (wrist_y_diff) ===")
print(df["wrist_y_diff"].describe())

print("\n=== 스무딩된 손목 높이 차이 ===")
if "wrist_y_diff_smooth" in df.columns:
    print(df["wrist_y_diff_smooth"].describe())

print("\n=== 바벨 기울기 (신뢰 가능한 것만) ===")
valid_angle = df["bar_angle"].dropna()
print(f"유효 프레임: {len(valid_angle)}/{len(df)}")
print(valid_angle.describe())

print("\n=== 손목 간 거리 ===")
print(df["wrist_distance"].describe())

print("\n=== 높이 차이 상위 10개 (비틀림 큰 구간) ===")
if "wrist_y_diff_smooth" in df.columns:
    top = df.nlargest(10, "wrist_y_diff_smooth")
    for _, row in top.iterrows():
        print(f"  프레임 {int(row['frame'])} ({row['time']:.2f}초) - 높이차: {row['wrist_y_diff_smooth']:.1f}px, 거리: {row.get('wrist_distance', 0):.1f}px")
