"""Streamlit front-end: upload an ECoG .mat file, decode finger flexion, plot it."""

import io

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
import streamlit as st

from src.config import FS, NUM_FINGERS
from src.model_utils import available_subjects, load_subject_models
from src.prediction import predict_trajectory
from src.preprocessing import load_ecog_from_mat

matplotlib.use("Agg")

FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Little"]

# Categorical slots 1-5 (validated adjacent order) + chart chrome, light surface
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SURFACE = "#fcfcfb"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"

MAX_PLOT_POINTS = 8000  # decimate long recordings for display only


@st.cache_resource(show_spinner="Loading models...")
def get_models(subject):
    return load_subject_models(subject)


@st.cache_data(show_spinner=False)
def parse_upload(file_bytes):
    return load_ecog_from_mat(io.BytesIO(file_bytes))


@st.cache_data(show_spinner="Extracting features and decoding...")
def run_prediction(file_bytes, entry_idx, subject, smooth):
    ecog = parse_upload(file_bytes)[entry_idx]
    return predict_trajectory(ecog, get_models(subject), FS, smooth=smooth)


def plot_trajectories(pred, fs):
    """Small multiples: one panel per finger, shared time axis and y-scale."""
    n = pred.shape[0]
    stride = max(1, n // MAX_PLOT_POINTS)
    t = np.arange(n)[::stride] / fs
    y = pred[::stride]
    ylim = (np.nanmin(pred), np.nanmax(pred))
    pad = 0.05 * (ylim[1] - ylim[0] or 1.0)

    fig, axes = plt.subplots(NUM_FINGERS, 1, figsize=(11, 8.5), sharex=True,
                             facecolor=SURFACE)
    for i, ax in enumerate(axes):
        ax.set_facecolor(SURFACE)
        ax.plot(t, y[:, i], color=SERIES_COLORS[i], linewidth=1.5)
        ax.set_ylim(ylim[0] - pad, ylim[1] + pad)
        ax.set_title(f"Finger {i + 1} · {FINGER_NAMES[i]}", loc="left",
                     fontsize=10, color=INK_SECONDARY, pad=4)
        ax.grid(True, axis="y", color=GRIDLINE, linewidth=0.8)
        ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
    axes[-1].set_xlabel("Time (s)", color=INK_MUTED, fontsize=9)
    fig.supylabel("Predicted flexion", color=INK_MUTED, fontsize=9)
    fig.tight_layout()
    return fig


def main():
    st.set_page_config(page_title="ECoG Finger Decoder", page_icon="🧠", layout="wide")
    st.title("ECoG Finger Flexion Decoder")
    st.caption(
        "Upload a `.mat` file containing ECoG recordings (samples × channels) and the "
        "trained stacked XGBoost + Ridge ensemble will predict the flexion trajectory of "
        "all five fingers. BE 5210 final project — BCI Competition IV, dataset 4."
    )

    subjects = available_subjects()
    if not subjects:
        st.error(
            "No trained models found in `models/`. Run `python training/train.py` first "
            "(or add the trained model files to the image)."
        )
        st.stop()

    with st.sidebar:
        st.header("Settings")
        subject = st.selectbox(
            "Subject model", subjects, format_func=lambda s: f"Subject {s}",
            help="Models are subject-specific; the recording must have the same "
                 "channel count as that subject's training data.",
        )
        smooth = st.checkbox("Smooth predictions (Savitzky–Golay)", value=True)
        st.markdown("---")
        st.markdown(
            f"**Pipeline**  \n"
            f"CAR → 0.1–200 Hz bandpass → 60 Hz notch → "
            f"100 ms windows / 50 ms overlap → per-channel features → "
            f"3 × XGBoost + Ridge meta-learner → Akima upsampling  \n"
            f"Sampling rate: {FS} Hz"
        )

    uploaded = st.file_uploader("ECoG recording (.mat)", type=["mat"])
    if uploaded is None:
        st.info("Waiting for a `.mat` upload. Try the competition's `leaderboard_data.mat`.")
        return

    file_bytes = uploaded.getvalue()
    try:
        entries = parse_upload(file_bytes)
    except Exception as e:  # noqa: BLE001 - surface any parse failure to the user
        st.error(f"Could not read ECoG data from this file: {e}")
        return

    entry_idx = 0
    if len(entries) > 1:
        entry_idx = st.selectbox(
            "This file contains several recordings — which one to decode?",
            range(len(entries)),
            format_func=lambda i: f"Entry {i + 1}: {entries[i].shape[0]} samples × "
                                  f"{entries[i].shape[1]} channels",
            index=min(subject - 1, len(entries) - 1),
        )
    ecog = entries[entry_idx]
    st.write(f"Recording: **{ecog.shape[0]:,} samples × {ecog.shape[1]} channels** "
             f"({ecog.shape[0] / FS:.1f} s at {FS} Hz)")

    if not st.button("Decode finger movement", type="primary"):
        return

    try:
        pred = run_prediction(file_bytes, entry_idx, subject, smooth)
    except ValueError as e:
        st.error(str(e))
        return

    st.subheader("Predicted finger trajectories")
    st.pyplot(plot_trajectories(pred, FS), use_container_width=True)

    df = pd.DataFrame(pred, columns=[f"finger{i + 1}" for i in range(NUM_FINGERS)])
    df.insert(0, "time_s", np.arange(pred.shape[0]) / FS)

    with st.expander("Data table"):
        st.dataframe(df, use_container_width=True, height=300)

    col1, col2 = st.columns(2)
    col1.download_button("Download CSV", df.to_csv(index=False).encode(),
                         file_name=f"predicted_dg_subject{subject}.csv", mime="text/csv")
    buf = io.BytesIO()
    sio.savemat(buf, {"predicted_dg": pred}, do_compression=True)
    col2.download_button("Download .mat", buf.getvalue(),
                         file_name=f"predicted_dg_subject{subject}.mat",
                         mime="application/octet-stream")


if __name__ == "__main__":
    main()
