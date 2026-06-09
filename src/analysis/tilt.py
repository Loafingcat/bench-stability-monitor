"""손목 기반 비틀림 분석 (baseline 보정 + sticking + 신뢰도 붕괴)"""

import pandas as pd
import numpy as np

CONF_THRESHOLD = 0.4
BASELINE_CONF = 0.5      # baseline 계산용 (더 엄격)
BASELINE_TIME_LIMIT = 2.0  # 초반 N초 내에서 baseline 추출


def _compute_baseline(df: pd.DataFrame) -> float:
    """
    카메라 각도로 인한 고정 높이차(baseline offset) 계산.
    초반 안정 구간의 lockout(바벨 최대 높이) 상태 기준.
    """
    valid = df[(df["L_wrist_conf"] >= BASELINE_CONF) &
               (df["R_wrist_conf"] >= BASELINE_CONF) &
               (df["time"] <= BASELINE_TIME_LIMIT)].copy()

    if len(valid) < 5:
        # fallback: 전체 신뢰 프레임 중앙값
        valid = df[(df["L_wrist_conf"] >= BASELINE_CONF) &
                   (df["R_wrist_conf"] >= BASELINE_CONF)].copy()
        if len(valid) == 0:
            return 0.0
        valid["y_diff_signed"] = valid["L_wrist_y"] - valid["R_wrist_y"]
        return valid["y_diff_signed"].median()

    # 초반 구간에서 바벨이 가장 높은(=lockout) 프레임들
    lockout = valid.nsmallest(min(10, len(valid)), "R_wrist_y")
    baseline = (lockout["L_wrist_y"] - lockout["R_wrist_y"]).median()
    return baseline


def analyze_tilt(df: pd.DataFrame) -> dict:
    baseline = _compute_baseline(df)

    # --- baseline 보정 높이차 (신뢰 프레임만) ---
    valid = df[(df["L_wrist_conf"] >= CONF_THRESHOLD) &
               (df["R_wrist_conf"] >= CONF_THRESHOLD)].copy()
    # 부호 유지 높이차에서 baseline 빼고 절대값 = 진짜 비틀림량
    valid["y_diff_signed"] = valid["L_wrist_y"] - valid["R_wrist_y"]
    valid["tilt"] = (valid["y_diff_signed"] - baseline).abs()

    if len(valid) > 0:
        tilt_smooth = valid["tilt"].rolling(3, center=True, min_periods=1).mean()
        max_tilt = tilt_smooth.max()
        max_tilt_idx = tilt_smooth.idxmax()
        max_tilt_time = df.loc[max_tilt_idx, "time"]
        mean_tilt = tilt_smooth.mean()
    else:
        max_tilt = mean_tilt = 0
        max_tilt_time = 0

    # --- 바닥 정체(sticking): 최저점 근처 ---
    ry = df["R_wrist_y"].copy()
    ry_smooth = ry.rolling(3, center=True, min_periods=1).mean()
    velocity = ry_smooth.diff().abs()
    bottom_idx = ry_smooth.idxmax()
    bottom_y = ry_smooth.loc[bottom_idx]
    bottom_time = df.loc[bottom_idx, "time"]

    near_bottom = (bottom_y - ry_smooth).abs() < 30
    low_velocity = velocity < 3.0
    sticking_mask = near_bottom & low_velocity

    sticking_frames = df[sticking_mask]
    near = sticking_frames[(sticking_frames["time"] >= bottom_time - 0.5) &
                           (sticking_frames["time"] <= bottom_time + 0.5)]
    if len(near) > 0:
        sticking_start = near["time"].min()
        sticking_end = near["time"].max()
        sticking_duration = round(sticking_end - sticking_start, 2)
    else:
        sticking_start = sticking_end = bottom_time
        sticking_duration = 0

    # --- 신뢰도 붕괴 구간 ---
    conf_collapse = df[(df["L_wrist_conf"] < 0.3) & (df["R_wrist_conf"] >= 0.6)]
    if len(conf_collapse) > 0:
        collapse_start = conf_collapse["time"].min()
        collapse_end = conf_collapse["time"].max()
        collapse_duration = round(collapse_end - collapse_start, 2)
        collapse_frames = len(conf_collapse)
    else:
        collapse_start = collapse_duration = collapse_frames = 0

    return {
        "baseline_offset": round(baseline, 1),
        "mean_tilt": round(mean_tilt, 1),
        "max_tilt": round(max_tilt, 1),
        "max_tilt_time": round(max_tilt_time, 2),
        "valid_frame_ratio": round(len(valid) / len(df), 2),

        "sticking_duration": sticking_duration,
        "sticking_start": round(sticking_start, 2),
        "sticking_end": round(sticking_end, 2),
        "bottom_time": round(bottom_time, 2),

        "conf_collapse_duration": collapse_duration,
        "conf_collapse_start": round(collapse_start, 2) if collapse_frames else 0,
        "conf_collapse_frames": collapse_frames,
    }
