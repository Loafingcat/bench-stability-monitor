"""오른손목 궤적 이상 탐지 가능성 확인"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

df = pd.read_csv("data/outputs/bench_tracking.csv")

# 오른손은 신뢰도 높음 — conf 0.5 이상만, 8초 이전(거치 제외)
r = df[(df["R_wrist_conf"] >= 0.5) & (df["time"] <= 7.0)].copy().reset_index(drop=True)

# 스무딩
rx = r["R_wrist_x"].rolling(3, center=True, min_periods=1).mean()
ry = r["R_wrist_y"].rolling(3, center=True, min_periods=1).mean()

# 특징 추출
r["vx"] = rx.diff()                    # 수평 속도
r["vy"] = ry.diff()                    # 수직 속도
r["speed"] = np.sqrt(r["vx"]**2 + r["vy"]**2)
r["accel"] = r["speed"].diff().abs()   # 가속도(속도 변화)

# 수평 기준선(정상이면 거의 일정) 대비 이탈
baseline_x = rx.iloc[:30].mean()
r["x_deviation"] = (rx - baseline_x)

print("=== 오른손 궤적 특징 (시간대별) ===")
print("time   x이탈   수평속도  수직속도  가속도")
print("-" * 50)
for t in np.arange(2.0, 7.0, 0.33):
    seg = r[(r["time"] >= t) & (r["time"] < t + 0.33)]
    if len(seg) > 0:
        print(f"{t:.1f}s  {seg['x_deviation'].mean():+6.1f}  {seg['vx'].abs().mean():6.1f}   {seg['vy'].mean():+6.1f}   {seg['accel'].mean():6.1f}")

print()
print("=== 핵심 지표 최대값 발생 시점 ===")
for col, label in [("x_deviation", "수평이탈(절대)"), ("accel", "가속도"), ("speed", "순간속도")]:
    if col == "x_deviation":
        idx = r[col].abs().idxmax()
        val = r.loc[idx, col]
    else:
        idx = r[col].idxmax()
        val = r.loc[idx, col]
    print(f"  {label}: {val:+.1f} @ {r.loc[idx, 'time']:.2f}초")

# 수평 이탈이 4초대에 집중되는지
print()
print("=== 수평 이탈(x_deviation) 상위 8프레임 ===")
top = r.reindex(r["x_deviation"].abs().sort_values(ascending=False).index).head(8)
for _, row in top.iterrows():
    print(f"  {row['time']:.2f}초: x이탈 {row['x_deviation']:+.1f}px, 가속도 {row['accel']:.1f}")
