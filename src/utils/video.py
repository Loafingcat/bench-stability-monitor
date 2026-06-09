"""영상 로드 및 프레임 추출"""

import cv2
import numpy as np
from pathlib import Path


def load_video(video_path: str | Path) -> tuple[list[np.ndarray], int, tuple[int, int]]:
    """
    영상을 읽어서 프레임 리스트, fps, (width, height) 반환.

    Returns:
        frames: 프레임 리스트 (BGR numpy array)
        fps: 초당 프레임 수
        size: (width, height)
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise FileNotFoundError(f"영상을 열 수 없습니다: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    print(f"영상 로드 완료: {len(frames)}프레임, {fps}fps, {width}x{height}")
    print(f"영상 길이: {len(frames) / fps:.1f}초")

    return frames, fps, (width, height)


def frame_to_time(frame_idx: int, fps: int) -> float:
    """프레임 번호 → 초 단위 시간"""
    return frame_idx / fps


def save_frame(frame: np.ndarray, path: str | Path) -> None:
    """프레임 1장을 이미지로 저장"""
    cv2.imwrite(str(path), frame)
