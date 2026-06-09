import streamlit as st
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.video import load_video
from src.detection.plate_tracker import track_all_frames, save_tracking_csv
from src.analysis.tilt import analyze_tilt
from src.analysis.trajectory import analyze_trajectory
from src.engine.event_detector import detect_events
from src.engine.risk_scorer import calculate_risk_score
from src.visualization.overlay_video import create_overlay_video

st.set_page_config(
    page_title="Bench Stability Monitor",
    layout="wide"
)

st.title("Bench Stability Monitor")

uploaded = st.file_uploader(
    "Upload Video",
    type=["mp4", "avi", "mov"]
)

if uploaded:

    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    video_path = upload_dir / uploaded.name
    video_path.write_bytes(uploaded.read())

    output_dir = Path("data/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    tracking_csv = output_dir / "tracking.csv"
    overlay_path = output_dir / "bench_analyzed.mp4"

    with st.spinner("Loading video..."):

        frames, fps, size = load_video(str(video_path))

        st.write(f"Frames: {len(frames)}")
        st.write(f"FPS: {fps}")
        st.write(f"Size: {size}")

    with st.spinner("Tracking..."):

        df = track_all_frames(
            frames,
            fps
        )

    with st.spinner("Analyzing..."):

        tilt = analyze_tilt(df)

        traj = analyze_trajectory(
            df,
            lift_end=tilt.get(
                "lift_end_time",
                7.0
            )
        )

        events = detect_events(
            tilt,
            traj
        )

        risk = calculate_risk_score(
            tilt,
            traj
        )

    save_tracking_csv(
        df,
        tracking_csv
    )

    with st.spinner("Generating overlay..."):

        create_overlay_video(
            video_path=video_path,
            tracking_csv=tracking_csv,
            output_path=overlay_path,
            trajectory_result=traj,
            tilt_result=tilt,
            risk_result=risk,
            events=events
        )

    st.success("Finished")

    st.subheader("Analysis Result")

    st.json(
        {
            "risk_score": risk["score"],
            "grade": risk["grade"],
            "max_deviation": traj["max_deviation"],
            "twist_duration": traj["twist_duration"]
        }
    )

    st.write(f"overlay exists: {overlay_path.exists()}")

    if overlay_path.exists():

        file_size = overlay_path.stat().st_size / 1024 / 1024

        st.write(
            f"overlay size: {file_size:.2f} MB"
        )

        st.write(
            f"path: {overlay_path.absolute()}"
        )

        st.divider()

        st.subheader("Analyzed Video")

        try:

            video_bytes = overlay_path.read_bytes()

            st.video(video_bytes)

            st.success("Video loaded successfully")

        except Exception as e:

            st.error(
                f"Video loading failed: {e}"
            )

        with open(overlay_path, "rb") as f:

            st.download_button(
                label="Download Overlay Video",
                data=f,
                file_name="bench_analyzed.mp4",
                mime="video/mp4"
            )

    else:

        st.error(
            "Overlay video not found."
        )