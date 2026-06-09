"""종합 위험도 점수 산출 (3대 지표 기반)"""


def _normalize(value: float, low: float, high: float) -> float:
    clamped = max(low, min(value, high))
    return ((clamped - low) / (high - low)) * 100


def calculate_risk_score(tilt_result: dict, trajectory_result: dict) -> dict:
    """
    3대 지표로 종합 위험도 산출.

    1. 수평 이탈 (trajectory) — 가중치 0.5: 비틀림의 직접 증거
    2. 바닥 정체 (sticking)  — 가중치 0.25: 실패/부상 위험 시간
    3. 신뢰도 붕괴 (conf)    — 가중치 0.25: 자세 비대칭 정황
    """
    # 1. 수평 이탈: 30px 이하 정상, 80px+ 위험
    deviation_score = _normalize(
        trajectory_result["max_deviation"], low=30, high=80
    )

    # 2. 바닥 정체: 0.2초 이하 정상, 0.8초+ 위험
    sticking_score = _normalize(
        tilt_result["sticking_duration"], low=0.2, high=0.8
    )

    # 3. 신뢰도 붕괴: 0프레임 정상, 30프레임+ 위험
    collapse_score = _normalize(
        tilt_result["conf_collapse_frames"], low=0, high=30
    )

    # 가중합
    score = (
        0.50 * deviation_score
        + 0.25 * sticking_score
        + 0.25 * collapse_score
    )
    score = round(min(score, 100), 1)

    # 등급
    if score >= 70:
        grade = "위험"
    elif score >= 40:
        grade = "주의"
    else:
        grade = "안정"

    return {
        "score": score,
        "grade": grade,
        "breakdown": {
            "수평이탈": round(deviation_score, 1),
            "바닥정체": round(sticking_score, 1),
            "신뢰도붕괴": round(collapse_score, 1),
        },
    }
