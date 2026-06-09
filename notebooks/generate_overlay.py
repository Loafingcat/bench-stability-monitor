"""오버레이 영상 생성"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.analysis.tilt import analyze_tilt
from src.analysis.trajectory import analyze_trajectory
from src.engine.event_detector import detect_events
from src.engine.risk_scorer import calculate_risk_score
from src.visualization.overlay_video import create_overlay_video

df = pd.read_csv("data/outputs/bench_tracking.csv")

tilt = analyze_tilt(df)
traj = analyze_trajectory(df)
events = detect_events(tilt, traj)
risk = calculate_risk_score(tilt, traj)

print("분석 완료, 오버레이 영상 생성 시작...")

create_overlay_video(
    video_path="data/videos/bench.mp4",
    tracking_csv="data/outputs/bench_tracking.csv",
    output_path="data/outputs/bench_analyzed.mp4",
    trajectory_result=traj,
    tilt_result=tilt,
    risk_result=risk,
    events=events,
)

print("완료! data/outputs/bench_analyzed.mp4 재생해봐")
