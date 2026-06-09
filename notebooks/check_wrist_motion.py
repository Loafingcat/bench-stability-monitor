"""4초 구간 전후 양쪽 손목 수직 움직임 확인"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

df = pd.read_csv("data/outputs/bench_tracking.csv")

# 3.5초 ~ 6.5초 구간 (프레임 105~195)
window = df[(df["frame"] >= 105) & (df["frame"] <= 195)]

print("frame  time   L_wrist_y  R_wrist_y  L_conf  R_conf")
print("-" * 60)
for _, row in window.iterrows():
    ly = row["L_wrist_y"]
    ry = row["R_wrist_y"]
    lc = row["L_wrist_conf"]
    rc = row["R_wrist_conf"]
    ly_s = f"{ly:.0f}" if pd.notna(ly) else "  --"
    ry_s = f"{ry:.0f}" if pd.notna(ry) else "  --"
    print(f"{int(row['frame']):4d}  {row['time']:.2f}   {ly_s:>7}   {ry_s:>7}   {lc:.2f}   {rc:.2f}")
