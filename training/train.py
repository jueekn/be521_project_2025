"""Train the per-subject, per-finger stacked XGBoost + Ridge ensembles.

Not used by the Streamlit app. Run once to populate ``models/``::

    python training/train.py --data raw_training_data.mat --device cuda

The training data is the BCI Competition IV dataset 4 file with ``train_ecog``
and ``train_dg`` cell arrays (one entry per subject).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from xgboost import XGBRegressor

# Allow running as a script from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import FS, MODELS_DIR, NUM_FINGERS, NUM_SUBJECTS  # noqa: E402
from src.features import build_design_matrix, downsample  # noqa: E402
from src.model_utils import save_finger_model  # noqa: E402


def make_base_models(device):
    common = dict(n_estimators=400, tree_method='hist', device=device)
    xgb1 = XGBRegressor(max_depth=4, learning_rate=0.03, subsample=0.85,
                        colsample_bytree=0.9, **common)
    xgb2 = XGBRegressor(max_depth=5, learning_rate=0.02, subsample=0.8,
                        colsample_bytree=0.85, **common)
    xgb3 = XGBRegressor(max_depth=5, learning_rate=0.02, subsample=0.8, **common)
    return xgb1, xgb2, xgb3


def train_finger(X_train, y_train, X_val, y_val, device):
    base_models = make_base_models(device)
    for xgb in base_models:
        xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Train Meta Ridge on validation-set predictions of the base models
    val_preds = np.stack([m.predict(X_val) for m in base_models], axis=1)
    ridge_meta = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    ridge_meta.fit(val_preds, y_val)

    final_preds = ridge_meta.predict(val_preds)
    r, _ = pearsonr(y_val, final_preds)
    return base_models, ridge_meta, r


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="raw_training_data.mat",
                        help="Path to raw_training_data.mat")
    parser.add_argument("--models-dir", default=str(MODELS_DIR),
                        help="Where to write trained models")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                        help="XGBoost device")
    parser.add_argument("--val-frac", type=float, default=0.05,
                        help="Fraction of windows held out for the meta-learner")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)

    proj_data = sio.loadmat(args.data)
    data_glove = proj_data['train_dg'].flatten()
    ecog = proj_data['train_ecog'].flatten()

    correlations = []

    for subj_idx in range(NUM_SUBJECTS):
        subject = subj_idx + 1
        print(f"\n Subject {subject} Training & Evaluation")

        # Feature extraction
        R = build_design_matrix(ecog[subj_idx], FS)
        glove = downsample(data_glove[subj_idx], num_windows=R.shape[0])

        # Split
        split_idx = int((1 - args.val_frac) * len(R))
        X_train, X_val = R[:split_idx], R[split_idx:]
        y_train, y_val = glove[:split_idx], glove[split_idx:]

        subject_corr = []
        for f in range(NUM_FINGERS):
            finger = f + 1
            print(f"Finger {finger}")
            base_models, ridge_meta, r = train_finger(
                X_train, y_train[:, f], X_val, y_val[:, f], args.device)
            print(f"Correlation: {r:.4f}")
            subject_corr.append(r)
            save_finger_model(subject, finger, base_models, ridge_meta, models_dir)

        avg_corr = np.mean([r for i, r in enumerate(subject_corr) if i != 3])
        print(f"Average correlation (excluding Finger 4): {avg_corr:.4f}")
        correlations.append(subject_corr)

    # Final avg
    total_avg = sum(r for sub in correlations for i, r in enumerate(sub) if i != 3) / 12
    print(f"\n Overall Average Correlation (excluding Finger 4): {total_avg:.4f}")
    print(f"Models saved under {models_dir}")


if __name__ == "__main__":
    main()
