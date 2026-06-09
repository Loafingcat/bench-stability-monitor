"""전체 분석 파이프라인"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.analysis.tilt import analyze_tilt
from src.analysis.trajectory import analyze_trajectory
from src.engine.event_detector import detect_events
from src.engine.risk_scorer import calculate_risk_score

df = pd.read_csv("data/outputs/bench_tracking.csv")

print("=" * 55)
print("  Bench Stability Monitor — 분석 결과")
print("=" * 55)

# 분석
tilt = analyze_tilt(df)
traj = analyze_trajectory(df, lift_end=tilt.get("lift_end_time", 7.0))

print(f"\n영상 정보: 293프레임, 30fps, 9.8초")
print(f"유효 분석 구간: 0 ~ {tilt.get('lift_end_time', 7.0)}초")
print(f"카메라 baseline offset: {tilt['baseline_offset']}px")

print(f"\n--- 궤적 분석 ---")
print(f"최대 수평 이탈: {traj['max_deviation']}px ({traj['max_deviation_time']}초)")
print(f"위험 구간(60px+): {traj['twist_start']}~{traj['twist_end']}초 ({traj['twist_duration']}초간)")

print(f"\n--- 리프트 분석 ---")
print(f"가슴 도달(최저점): {tilt['bottom_time']}초")
print(f"바닥 정체: {tilt['sticking_duration']}초 ({tilt['sticking_start']}~{tilt['sticking_end']}초)")

print(f"\n--- 비대칭 분석 ---")
print(f"왼손목 검출 실패: {tilt['conf_collapse_frames']}프레임 ({tilt['conf_collapse_duration']}초)")

# 위험 이벤트
events = detect_events(tilt, traj)
print(f"\n{'=' * 55}")
print(f"  위험 이벤트: {len(events)}건")
print(f"{'=' * 55}")
for e in events:
    icon = "🔴" if e["severity"] == "위험" else "🟡"
    print(f"  {icon} {e['time']}초 [{e['type']}] {e['detail']}")

# 종합 점수
risk = calculate_risk_score(tilt, traj)
print(f"\n{'=' * 55}")
grade_icon = {"안정": "🟢", "주의": "🟡", "위험": "🔴"}
print(f"  종합 위험도: {risk['score']}점 {grade_icon[risk['grade']]} {risk['grade']}")
print(f"{'=' * 55}")
print(f"  수평이탈  {risk['breakdown']['수평이탈']:5.1f}점 (가중치 50%)")
print(f"  바닥정체  {risk['breakdown']['바닥정체']:5.1f}점 (가중치 25%)")
print(f"  신뢰도붕괴 {risk['breakdown']['신뢰도붕괴']:5.1f}점 (가중치 25%)")
print(f"{'=' * 55}")
