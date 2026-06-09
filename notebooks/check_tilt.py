import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.analysis.tilt import analyze_tilt

df = pd.read_csv("data/outputs/bench_tracking.csv")
r = analyze_tilt(df)

print("=== 비틀림 분석 (baseline 보정) ===")
print(f"baseline offset (카메라 각도): {r['baseline_offset']}px")
print(f"신뢰 가능 프레임 비율: {r['valid_frame_ratio']:.0%}")
print()
print(f"비틀림(baseline 대비) - 평균 {r['mean_tilt']}px, 최대 {r['max_tilt']}px ({r['max_tilt_time']}초)")
print()
print(f"[최저점/가슴 찍는 순간] {r['bottom_time']}초")
print(f"[바닥 정체] {r['sticking_duration']}초 ({r['sticking_start']}~{r['sticking_end']}초)")
print(f"[신뢰도 붕괴] {r['conf_collapse_duration']}초 ({r['conf_collapse_start']}초부터, {r['conf_collapse_frames']}프레임)")
