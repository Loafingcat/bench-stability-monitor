"""YOLOv8-Pose 기반 관절 키포인트 검출"""

from ultralytics import YOLO
import numpy as np

KEYPOINTS = {
    'nose': 0,
    'L_shoulder': 5, 'R_shoulder': 6,
    'L_elbow': 7, 'R_elbow': 8,
    'L_wrist': 9, 'R_wrist': 10,
    'L_hip': 11, 'R_hip': 12,
}

MIN_WRIST_DISTANCE = 30


class PoseDetector:
    def __init__(self, model_path='yolov8m-pose.pt', conf=0.3):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect_frame(self, frame):
        results = self.model(frame, conf=self.conf, verbose=False)
        for r in results:
            if r.keypoints is None or len(r.keypoints) == 0:
                return None
            kpts = r.keypoints[0].data[0]
            output = {}
            for name, idx in KEYPOINTS.items():
                x, y, c = kpts[idx].tolist()
                output[name] = (x, y, c)

            lw = output['L_wrist']
            rw = output['R_wrist']
            if lw[2] > 0.3 and rw[2] > 0.3:
                cx = (lw[0] + rw[0]) / 2
                cy = (lw[1] + rw[1]) / 2
                output['bar_center'] = (cx, cy)
                dx = rw[0] - lw[0]
                dy = rw[1] - lw[1]
                wrist_dist = np.sqrt(dx**2 + dy**2)
                output['wrist_distance'] = wrist_dist
                output['wrist_y_diff'] = abs(rw[1] - lw[1])
                if wrist_dist >= MIN_WRIST_DISTANCE:
                    output['bar_angle'] = np.degrees(np.arctan2(abs(dy), abs(dx)))
                else:
                    output['bar_angle'] = None
            else:
                output['bar_center'] = None
                output['wrist_distance'] = None
                output['wrist_y_diff'] = None
                output['bar_angle'] = None
            return output
        return None
