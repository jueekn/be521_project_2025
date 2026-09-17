"""Inference: raw ECoG -> finger flexion trajectories at the original sampling rate."""

import numpy as np
from scipy.interpolate import Akima1DInterpolator
from scipy.signal import savgol_filter

from src.config import FS, NUM_FINGERS, WINDOW_LENGTH, WINDOW_OVERLAP
from src.features import build_design_matrix
from src.model_utils import expected_num_features


def upsample_predictions(y_pred_downsampled, original_length, window_length=WINDOW_LENGTH,
                         window_overlap=WINDOW_OVERLAP, fs=FS, smooth=True):
    y_pred_upsampled = []
    step_size = int((window_length - window_overlap) * fs)
    full_time = np.arange(original_length)

    for subject_data in y_pred_downsampled:
        num_windows = subject_data.shape[0]
        time_pred = np.arange(0, num_windows * step_size, step_size)
        if time_pred[-1] < original_length - 1:
            time_pred = np.append(time_pred, original_length - 1)
            subject_data = np.vstack([subject_data, subject_data[-1]])

        upsampled_data = np.vstack([
            Akima1DInterpolator(time_pred, subject_data[:, i])(full_time)
            for i in range(subject_data.shape[1])
        ]).T

        if smooth:
            for i in range(upsampled_data.shape[1]):
                signal = upsampled_data[:, i]

                # Adaptive smoothing
                local_std = np.std(signal)
                if local_std < 0.02:
                    signal = savgol_filter(signal, window_length=51, polyorder=2, mode='interp')
                else:
                    signal = savgol_filter(signal, window_length=11, polyorder=2, mode='interp')

                upsampled_data[:, i] = signal

        y_pred_upsampled.append(upsampled_data)

    return y_pred_upsampled


def predict_finger(finger_model, R):
    """Stacked ensemble prediction for one finger on an R matrix."""
    base_preds = np.stack([m.predict(R) for m in finger_model["base_models"]], axis=1)
    return finger_model["meta_model"].predict(base_preds)


def predict_windows(subject_models, R):
    """Downsampled (one row per window) predictions for all fingers."""
    n_expected = expected_num_features(subject_models)
    if R.shape[1] != n_expected:
        raise ValueError(
            f"Feature dimension mismatch: data produced {R.shape[1]} features but the "
            f"selected models expect {n_expected}. Check that the recording's channel "
            f"count matches the subject these models were trained on."
        )
    Y = np.zeros((R.shape[0], NUM_FINGERS))
    for j, finger_model in enumerate(subject_models):
        Y[:, j] = predict_finger(finger_model, R)
    return Y


def predict_trajectory(raw_ecog, subject_models, fs=FS, smooth=True):
    """End-to-end: raw ECoG (samples, channels) -> flexion (samples, 5)."""
    R = build_design_matrix(raw_ecog, fs)
    Y_windows = predict_windows(subject_models, R)
    return upsample_predictions([Y_windows], raw_ecog.shape[0], smooth=smooth)[0]
