import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.analysis.trajectory import analyze_trajectory

df = pd.read_csv("data/outputs/bench_tracking.csv")
r = analyze_trajectory(df)

print("=== 오른손 궤적 비틀림 분석 ===")
print(f"수평 기준선: {r['baseline_x']}px")
print(f"최대 수평 이탈: {r['max_deviation']}px ({r['max_deviation_time']}초)")
print(f"최대 이탈 가속: {r['max_drift_rate']}px/frame ({r['max_drift_time']}초)")
print()
print(f"[주의 시작] 이탈 40px+ → {r['warning_start']}초")
print(f"[위험 구간] 이탈 60px+ → {r['twist_start']}~{r['twist_end']}초 ({r['twist_duration']}초간, {r['twist_frames']}프레임)")
