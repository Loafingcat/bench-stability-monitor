"""전체 프레임에 대해 키포인트 추적 + CSV 저장"""

import pandas as pd
import numpy as np
from pathlib import Path
from src.detection.detector import PoseDetector


def track_all_frames(frames: list[np.ndarray], fps: int) -> pd.DataFrame:
    detector = PoseDetector()
    records = []

    keypoint_names = [
        "L_wrist", "R_wrist",
        "L_shoulder", "R_shoulder",
        "L_elbow", "R_elbow",
        "L_hip", "R_hip",
    ]

    for i, frame in enumerate(frames):
        row = {"frame": i, "time": i / fps}
        result = detector.detect_frame(frame)

        if result is None:
            for name in keypoint_names:
                row[f"{name}_x"] = np.nan
                row[f"{name}_y"] = np.nan
                row[f"{name}_conf"] = 0.0
            row["bar_center_x"] = np.nan
            row["bar_center_y"] = np.nan
            row["bar_angle"] = np.nan
            row["wrist_y_diff"] = np.nan
            row["wrist_distance"] = np.nan
        else:
            for name in keypoint_names:
                x, y, c = result[name]
                row[f"{name}_x"] = x if c > 0.3 else np.nan
                row[f"{name}_y"] = y if c > 0.3 else np.nan
                row[f"{name}_conf"] = c

            if result["bar_center"] is not None:
                row["bar_center_x"] = result["bar_center"][0]
                row["bar_center_y"] = result["bar_center"][1]
            else:
                row["bar_center_x"] = np.nan
                row["bar_center_y"] = np.nan

            row["bar_angle"] = result.get("bar_angle")  # None이면 NaN으로
            row["wrist_y_diff"] = result.get("wrist_y_diff")
            row["wrist_distance"] = result.get("wrist_distance")

        records.append(row)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(frames)} 프레임 처리 완료")

    df = pd.DataFrame(records)

    # 보간 + 스무딩
    cols_to_interpolate = [c for c in df.columns if c.endswith(("_x", "_y")) and "conf" not in c]
    cols_to_interpolate += ["bar_angle", "wrist_y_diff", "wrist_distance"]
    df[cols_to_interpolate] = df[cols_to_interpolate].interpolate(method="linear", limit=5)

    # 이동 평균 스무딩 (3프레임 윈도우)
    smooth_cols = ["bar_angle", "wrist_y_diff", "bar_center_x", "bar_center_y"]
    for col in smooth_cols:
        if col in df.columns:
            df[f"{col}_smooth"] = df[col].rolling(window=3, center=True, min_periods=1).mean()

    return df


def save_tracking_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"저장 완료: {path} ({len(df)}행)")
