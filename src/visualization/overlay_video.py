"""Overlay analysis on video"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def create_overlay_video(
    video_path,
    tracking_csv,
    output_path,
    trajectory_result,
    tilt_result,
    risk_result,
    events,
):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if w <= 0 or h <= 0:
        raise RuntimeError(
            f"Invalid video size: width={w}, height={h}"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (w, h),
    )

    print(f"video: {video_path}")
    print(f"fps: {fps}")
    print(f"size: {w} x {h}")
    print(f"writer: {out.isOpened()}")

    if not out.isOpened():
        raise RuntimeError(
            f"VideoWriter open failed: {output_path}"
        )

    df = pd.read_csv(tracking_csv)

    print(f"rows: {len(df)}")
    print(f"trajectory: {trajectory_result}")

    r_trail = []
    l_trail = []

    C_SAFE = (0, 200, 0)
    C_WARN = (0, 200, 255)
    C_DANGER = (0, 0, 255)

    C_SKEL = (255, 180, 0)
    C_TR = (255, 100, 0)
    C_TL = (0, 100, 255)

    tw_s = float(
        trajectory_result.get("twist_start", 0)
    )

    tw_e = float(
        trajectory_result.get("twist_end", 0)
    )

    w_s = float(
        trajectory_result.get("warning_start", 0)
    )

    bx = trajectory_result.get("baseline_x")

    if bx is None or pd.isna(bx):
        bx = w // 2

    fi = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if fi >= len(df):
            out.write(frame)
            fi += 1
            continue

        row = df.iloc[fi]

        t = float(row["time"])

        ov = frame.copy()

        if tw_s <= t <= tw_e:
            sc = C_DANGER
            st = "DANGER"

        elif t >= w_s and t < tw_s and w_s > 0:
            sc = C_WARN
            st = "WARNING"

        else:
            sc = C_SAFE
            st = "STABLE"

        if st == "DANGER":
            cv2.rectangle(
                ov,
                (0, 0),
                (w - 1, h - 1),
                C_DANGER,
                12,
            )

        kps = {}

        for nm in [
            "L_shoulder",
            "R_shoulder",
            "L_elbow",
            "R_elbow",
            "L_wrist",
            "R_wrist",
        ]:
            xv = row.get(f"{nm}_x")
            yv = row.get(f"{nm}_y")
            cf = row.get(f"{nm}_conf", 0)

            if (
                pd.notna(xv)
                and pd.notna(yv)
                and cf > 0.3
            ):
                kps[nm] = (
                    int(xv),
                    int(yv),
                    cf,
                )

        for nm, (x, y, cf) in kps.items():
            color = (
                C_SKEL
                if cf >= 0.5
                else (100, 100, 100)
            )

            radius = 8 if cf >= 0.5 else 5

            cv2.circle(
                ov,
                (x, y),
                radius,
                color,
                -1,
            )

        connections = [
            ("L_shoulder", "R_shoulder"),
            ("L_shoulder", "L_elbow"),
            ("L_elbow", "L_wrist"),
            ("R_shoulder", "R_elbow"),
            ("R_elbow", "R_wrist"),
            ("L_wrist", "R_wrist"),
        ]

        for a, b in connections:
            if a in kps and b in kps:
                if a == "L_wrist" and b == "R_wrist":
                    color = sc
                    thickness = 4
                else:
                    color = C_SKEL
                    thickness = 2

                cv2.line(
                    ov,
                    (kps[a][0], kps[a][1]),
                    (kps[b][0], kps[b][1]),
                    color,
                    thickness,
                )

        if "R_wrist" in kps:
            r_trail.append(
                (
                    kps["R_wrist"][0],
                    kps["R_wrist"][1],
                    t,
                )
            )

        if "L_wrist" in kps:
            l_trail.append(
                (
                    kps["L_wrist"][0],
                    kps["L_wrist"][1],
                    t,
                )
            )

        r_trail = [
            (x, y, tt)
            for x, y, tt in r_trail
            if t - tt < 0.5
        ]

        l_trail = [
            (x, y, tt)
            for x, y, tt in l_trail
            if t - tt < 0.5
        ]

        for i in range(1, len(r_trail)):
            cv2.line(
                ov,
                (r_trail[i - 1][0], r_trail[i - 1][1]),
                (r_trail[i][0], r_trail[i][1]),
                C_TR,
                max(2, int(i / len(r_trail) * 5)),
            )

        for i in range(1, len(l_trail)):
            cv2.line(
                ov,
                (l_trail[i - 1][0], l_trail[i - 1][1]),
                (l_trail[i][0], l_trail[i][1]),
                C_TL,
                max(2, int(i / len(l_trail) * 5)),
            )

        bxi = int(bx)

        cv2.line(
            ov,
            (bxi, 800),
            (bxi, h),
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        cur_dev = (
            abs(kps["R_wrist"][0] - bx)
            if "R_wrist" in kps
            else 0
        )

        cv2.rectangle(
            ov,
            (0, 0),
            (w, 160),
            (30, 30, 30),
            -1,
        )

        cv2.putText(
            ov,
            f"{t:.2f}s",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            3,
        )

        cv2.putText(
            ov,
            st,
            (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            sc,
            3,
        )

        cv2.putText(
            ov,
            f"Risk: {risk_result['score']:.0f}",
            (30, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            sc,
            2,
        )

        frame = cv2.addWeighted(
            ov,
            0.85,
            frame,
            0.15,
            0,
        )

        out.write(frame)

        fi += 1

        if fi % 50 == 0:
            print(f"{fi}/{len(df)} done")

    cap.release()
    out.release()

    print(f"saved: {output_path}")