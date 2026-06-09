"""Right wrist trajectory-based twist detection"""

import pandas as pd
import numpy as np

R_CONF_THRESHOLD = 0.5
LIFT_END_DEFAULT = 7.0
BASELINE_START = 2.0
BASELINE_END = 3.0


def analyze_trajectory(df, lift_end=LIFT_END_DEFAULT):
    r = df[(df['R_wrist_conf'] >= R_CONF_THRESHOLD) &
           (df['time'] <= lift_end)].copy().reset_index(drop=True)
    if len(r) < 10:
        return {'error': 'not enough data'}

    rx = r['R_wrist_x'].rolling(3, center=True, min_periods=1).mean()
    ry = r['R_wrist_y'].rolling(3, center=True, min_periods=1).mean()

    bl = r[(r['time'] >= BASELINE_START) & (r['time'] <= BASELINE_END)]
    if len(bl) >= 5:
        baseline_x = rx[bl.index[0]:bl.index[-1]+1].mean()
    else:
        baseline_x = rx.iloc[:30].mean()

    x_deviation = rx - baseline_x
    r['x_dev'] = x_deviation
    r['x_dev_abs'] = x_deviation.abs()

    dev_velocity = x_deviation.diff().rolling(3, center=True, min_periods=1).mean()
    r['dev_velocity'] = dev_velocity

    analysis_mask = r['time'] >= BASELINE_START
    r_analysis = r[analysis_mask]

    if len(r_analysis) == 0:
        return {'error': 'no frames after baseline'}

    max_idx = r_analysis['x_dev_abs'].idxmax()
    max_dev = r.loc[max_idx, 'x_dev']
    max_dev_time = r.loc[max_idx, 'time']

    max_accel_idx = dev_velocity[analysis_mask].abs().idxmax()
    max_drift_rate = dev_velocity.loc[max_accel_idx]
    max_drift_time = r.loc[max_accel_idx, 'time']

    CRITICAL_THRESHOLD = 60.0
    twist_zone = r_analysis[r_analysis['x_dev_abs'] >= CRITICAL_THRESHOLD]
    if len(twist_zone) > 0:
        twist_start = twist_zone['time'].min()
        twist_end = twist_zone['time'].max()
        twist_duration = round(twist_end - twist_start, 2)
        twist_frames = len(twist_zone)
    else:
        twist_start = twist_end = twist_duration = 0
        twist_frames = 0

    WARNING_THRESHOLD = 40.0
    warn_zone = r_analysis[r_analysis['x_dev_abs'] >= WARNING_THRESHOLD]
    warn_start = warn_zone['time'].min() if len(warn_zone) > 0 else 0

    return {
        'baseline_x': round(baseline_x, 1),
        'max_deviation': round(abs(max_dev), 1),
        'max_deviation_time': round(max_dev_time, 2),
        'max_drift_rate': round(abs(max_drift_rate), 1),
        'max_drift_time': round(max_drift_time, 2),
        'warning_start': round(warn_start, 2),
        'twist_start': round(twist_start, 2),
        'twist_end': round(twist_end, 2),
        'twist_duration': twist_duration,
        'twist_frames': twist_frames,
        'deviation_series': x_deviation,
        'time_series': r['time'],
    }
