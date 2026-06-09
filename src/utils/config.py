"""프로젝트 전체 설정값"""

from pathlib import Path

# 경로
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
VIDEO_DIR = DATA_DIR / "videos"
OUTPUT_DIR = DATA_DIR / "outputs"

# 바벨 검출
DETECTION_METHOD = "yolo"  # "yolo" or "opencv"
YOLO_MODEL_PATH = MODELS_DIR / "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.5

# 분석 임계값
TILT_WARNING_DEG = 5.0
TILT_DANGER_DEG = 8.0
TILT_SPIKE_DEG = 3.0

SHAKE_WARNING = 15.0
SHAKE_DANGER = 30.0

VELOCITY_DROP_RATIO = 0.5

# 위험도 가중치
RISK_WEIGHTS = {
    "tilt": 0.5,
    "shake": 0.2,
    "velocity": 0.2,
    "trajectory": 0.1,
}

# 등급 기준
GRADE_THRESHOLDS = {
    "안정": (0, 40),
    "주의": (40, 70),
    "위험": (70, 100),
}
