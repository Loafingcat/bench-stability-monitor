"""오른손목 y → 왼손목 y 관계 분석 (회귀 가능성 확인)"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

df = pd.read_csv("data/outputs/bench_tracking.csv")

# 양쪽 모두 신뢰도 높은 프레임만 (학습 데이터)
valid = df[(df["L_wrist_conf"] >= 0.5) & (df["R_wrist_conf"] >= 0.6)].copy()
print(f"신뢰 프레임 수: {len(valid)}")

x = valid["R_wrist_y"].values
y = valid["L_wrist_y"].values

# 1차 선형 회귀
coef = np.polyfit(x, y, 1)
slope, intercept = coef
pred = slope * x + intercept
residuals = y - pred

# 상관계수, 오차
corr = np.corrcoef(x, y)[0, 1]
rmse = np.sqrt(np.mean(residuals**2))

print(f"\n=== 선형 관계 L_wrist_y = {slope:.3f} * R_wrist_y + {intercept:.1f} ===")
print(f"상관계수 r: {corr:.3f}")
print(f"RMSE (예측 오차): {rmse:.1f}px")
print(f"잔차 표준편차: {residuals.std():.1f}px")
print()

# 시간대별로 잔차가 어떤지 (부상 구간에서 튀는지)
valid["pred_L"] = pred
valid["residual"] = residuals
print("=== 시간대별 잔차 (실제 - 예측, 양수=왼손이 예측보다 아래=비틀림) ===")
for t_start in [0, 2, 4, 6, 8]:
    seg = valid[(valid["time"] >= t_start) & (valid["time"] < t_start + 2)]
    if len(seg) > 0:
        print(f"  {t_start}~{t_start+2}초: 평균잔차 {seg['residual'].mean():+.1f}px, 최대 {seg['residual'].abs().max():.1f}px (n={len(seg)})")

# 2차 다항식도 비교 (비선형이 더 맞을 수도)
coef2 = np.polyfit(x, y, 2)
pred2 = np.polyval(coef2, x)
rmse2 = np.sqrt(np.mean((y - pred2)**2))
print(f"\n2차 다항식 RMSE: {rmse2:.1f}px (1차보다 나으면 비선형 고려)")
