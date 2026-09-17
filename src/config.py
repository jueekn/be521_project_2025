"""Pipeline constants shared by training, prediction and the app.

These values must match between training and inference: the saved models
expect feature vectors built with exactly these settings.
"""

from pathlib import Path

# Sampling rate of the ECoG recordings (Hz)
FS = 1000

# Feature-window geometry (seconds)
WINDOW_LENGTH = 0.1
WINDOW_OVERLAP = 0.05

# Time lags (in windows) concatenated into each feature vector
LAGS = [0, 4]

# Number of windows stacked into each row of the R matrix
N_WIND = 3

# Causal shift applied to the data glove during training (ms)
GLOVE_SHIFT_MS = 37

# High-gamma bands used for Hilbert band power features (Hz)
BANDS = [(75, 115), (125, 159), (159, 175)]

NUM_SUBJECTS = 3
NUM_FINGERS = 5

# Repository layout
ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
