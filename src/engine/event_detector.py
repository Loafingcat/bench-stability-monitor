"""위험 이벤트 탐지 엔진 (3대 지표 기반)"""


def detect_events(tilt_result: dict, trajectory_result: dict) -> list[dict]:
    events = []

    # 1. 수평 이탈 위험 구간
    if trajectory_result.get("twist_duration", 0) > 0:
        events.append({
            "time": trajectory_result["twist_start"],
            "end_time": trajectory_result["twist_end"],
            "type": "수평이탈",
            "severity": "위험",
            "detail": f"최대 {trajectory_result['max_deviation']}px 이탈, "
                       f"{trajectory_result['twist_duration']}초간 지속",
        })

    # 2. 주의 시작
    if trajectory_result.get("warning_start", 0) > 0:
        events.append({
            "time": trajectory_result["warning_start"],
            "type": "이탈시작",
            "severity": "주의",
            "detail": f"수평 이탈 40px 초과 시작",
        })

    # 3. 바닥 정체 (sticking point)
    if tilt_result.get("sticking_duration", 0) > 0.1:
        events.append({
            "time": tilt_result["sticking_start"],
            "type": "바닥정체",
            "severity": "주의",
            "detail": f"{tilt_result['sticking_duration']}초간 정체 "
                       f"({tilt_result['sticking_start']}~{tilt_result['sticking_end']}초)",
        })

    # 4. 신뢰도 붕괴
    if tilt_result.get("conf_collapse_frames", 0) > 10:
        events.append({
            "time": tilt_result["conf_collapse_start"],
            "type": "자세비대칭",
            "severity": "위험",
            "detail": f"왼손목 검출 실패 {tilt_result['conf_collapse_frames']}프레임, "
                       f"{tilt_result['conf_collapse_duration']}초간",
        })

    events.sort(key=lambda e: e["time"])
    return events
