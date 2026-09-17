"""Signal preprocessing: loading, re-referencing and filtering raw ECoG."""

import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt, hilbert, iirnotch

from src.config import FS


def load_ecog_from_mat(source):
    """Load ECoG arrays from a .mat file (path or file-like object).

    Handles both the competition layouts used in this project:
      * a cell array of subjects (e.g. ``train_ecog`` / ``leaderboard_ecog``,
        shape (3, 1) with one (samples, channels) array per subject), and
      * a single 2D (samples, channels) array.

    Returns a list of float64 arrays, one per subject/entry found.
    """
    mat = sio.loadmat(source)
    arrays = []
    for key, value in mat.items():
        if key.startswith("__") or not isinstance(value, np.ndarray):
            continue
        if value.dtype == object:
            # MATLAB cell array -> one entry per subject
            for cell in value.flatten():
                if isinstance(cell, np.ndarray) and cell.ndim == 2 and cell.size > 0:
                    arrays.append(cell.astype(np.float64))
        elif value.ndim == 2 and value.size > 1 and np.issubdtype(value.dtype, np.number):
            arrays.append(value.astype(np.float64))
    if not arrays:
        raise ValueError("No 2D numeric ECoG arrays found in the .mat file.")
    return arrays


def apply_car(eeg):
    """Common average referencing"""
    return eeg - np.mean(eeg, axis=1, keepdims=True)


def filter_data(raw_eeg, fs=FS, lowcut=0.1, highcut=200, order=4):
    raw_eeg = apply_car(raw_eeg.astype(np.float64))

    # Bandpass filter
    nyq = 0.5 * fs
    b_band, a_band = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    eeg_bandpassed = filtfilt(b_band, a_band, raw_eeg, axis=0)

    # 60 Hz Notch filter
    notch_freq = 60.0
    Q = 40.0
    b_notch, a_notch = iirnotch(w0=notch_freq / nyq, Q=Q)
    eeg_filtered = filtfilt(b_notch, a_notch, eeg_bandpassed, axis=0)

    return eeg_filtered


def lmp(signal, window_len=100):
    """Local motor potential"""
    return np.convolve(signal, np.ones(window_len) / window_len, mode='same')


def hilbert_bandpower(signal, fs, low, high):
    nyq = 0.5 * fs
    b, a = butter(4, [low / nyq, high / nyq], btype='band')
    band = filtfilt(b, a, signal)
    envelope = np.abs(hilbert(band))
    return np.mean(envelope)
