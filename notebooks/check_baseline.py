"""시작 lockout 상태의 baseline 손목 높이차 측정"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

df = pd.read_csv("data/outputs/bench_tracking.csv")

# 양쪽 conf 모두 0.5 이상인 신뢰 프레임만
valid = df[(df["L_wrist_conf"] >= 0.5) & (df["R_wrist_conf"] >= 0.5)].copy()
valid["y_diff_signed"] = valid["L_wrist_y"] - valid["R_wrist_y"]  # 부호 유지

# lockout = 바벨 최대 높이 = R_wrist_y 최소값 근처
# 상위 (가장 높이 든) 15프레임
lockout = valid.nsmallest(15, "R_wrist_y")

print("=== Lockout 구간 (바벨 최대 높이) ===")
print(f"프레임 수: {len(lockout)}")
print(f"R_wrist_y 범위: {lockout['R_wrist_y'].min():.0f} ~ {lockout['R_wrist_y'].max():.0f}")
print()
print("이 구간 손목 높이차 (L - R, 부호 유지):")
print(f"  평균: {lockout['y_diff_signed'].mean():.1f}px")
print(f"  중앙값: {lockout['y_diff_signed'].median():.1f}px")
print(f"  표준편차: {lockout['y_diff_signed'].std():.1f}px")
print()
print(">> 이 값이 카메라 각도로 인한 baseline offset")
print()

# 초반 안정 구간(0~2초)의 lockout도 따로 확인
early = valid[valid["time"] <= 2.0]
if len(early) > 0:
    early_lockout = early.nsmallest(10, "R_wrist_y")
    print("=== 초반 2초 내 lockout ===")
    print(f"  손목 높이차 평균: {early_lockout['y_diff_signed'].mean():.1f}px")
    print(f"  시간 범위: {early_lockout['time'].min():.2f} ~ {early_lockout['time'].max():.2f}초")

# 전체 신뢰 프레임의 높이차 분포
print()
print("=== 전체 신뢰 프레임 높이차(부호) 분포 ===")
print(valid["y_diff_signed"].describe())
