"""Feature extraction: windowed per-channel features, time lags and the R matrix."""

import numpy as np
from antropy import spectral_entropy

from src.config import BANDS, FS, LAGS, N_WIND, WINDOW_LENGTH, WINDOW_OVERLAP, GLOVE_SHIFT_MS
from src.preprocessing import filter_data, hilbert_bandpower, lmp


def get_features(window, fs=FS):
    # Focused feature set: 6 time-domain + 3 high-gamma bands
    bands = BANDS
    num_channels = window.shape[1]
    # NOTE: 6 time-domain slots are allocated but only 5 are filled (kurtosis was
    # dropped), so the last column is always zero. Kept as-is so feature
    # dimensions match the original notebook / trained models.
    features = np.zeros((num_channels, 6 + len(bands)))  # 6 time + 3 freq

    for ch in range(num_channels):
        signal_ch = window[:, ch]
        features[ch, 0] = np.mean(signal_ch)                    # Mean
        features[ch, 1] = np.std(signal_ch)                     # Std
        features[ch, 2] = np.mean(np.diff(signal_ch))           # Derivative
        features[ch, 3] = spectral_entropy(signal_ch, sf=fs, normalize=True)  # Entropy
        features[ch, 4] = np.mean(lmp(signal_ch, window_len=100))             # LMP

        for i, (low, high) in enumerate(bands):
            features[ch, 5 + i] = hilbert_bandpower(signal_ch, fs, low, high)

    return features


def get_windowed_feats(raw_ecog, fs=FS, window_length=WINDOW_LENGTH,
                       window_overlap=WINDOW_OVERLAP, lags=LAGS):
    """
    Extracts features for each window, including time-lagged windows.
    lags: List of relative time steps to include (e.g., [0, 1] includes t and t-1)
    """
    window_length = int(window_length * fs)
    window_overlap = int(window_overlap * fs)
    step_size = window_length - window_overlap
    clean_data = filter_data(raw_ecog, fs)

    all_feats = []

    # Precompute features per window
    raw_feats = []
    for start in range(0, clean_data.shape[0] - window_length + 1, step_size):
        window = clean_data[start:start + window_length, :]
        feats = get_features(window, fs)
        raw_feats.append(feats.flatten())

    raw_feats = np.array(raw_feats)

    # Build feature matrix with time lags
    for i in range(max(lags), len(raw_feats)):
        lagged = [raw_feats[i - l] for l in lags]
        all_feats.append(np.concatenate(lagged))

    return np.array(all_feats)


def create_R_matrix(features, N_wind=N_WIND):
    M, num_feats = features.shape
    padded_features = np.vstack([np.tile(features[0], (N_wind - 1, 1)), features])
    R = np.ones((M, 1 + N_wind * num_feats))
    for t in range(M):
        R[t, 1:] = padded_features[t:t + N_wind].flatten()
    return R


def build_design_matrix(raw_ecog, fs=FS):
    """Full raw ECoG -> R matrix pipeline used by both training and inference."""
    feats = get_windowed_feats(raw_ecog, fs)
    return create_R_matrix(feats)


def downsample(flexion, num_windows, shift_ms=GLOVE_SHIFT_MS, fs=FS,
               window_len_ms=int(WINDOW_LENGTH * 1000),
               window_overlap_ms=int(WINDOW_OVERLAP * 1000)):
    """Downsample flexion with a -37 ms causal shift (training targets only)"""
    shift_samples = int((shift_ms / 1000.0) * fs)
    flexion = np.roll(flexion, -shift_samples, axis=0)
    effective_fs = fs / (window_len_ms - window_overlap_ms)
    downsample_factor = int(fs / effective_fs)
    downsampled = flexion[::downsample_factor]
    return downsampled[:num_windows]
