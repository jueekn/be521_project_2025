"""Saving and loading the per-subject / per-finger stacked ensembles.

On-disk layout::

    models/
      subject1/
        finger1/
          xgb1.json
          xgb2.json
          xgb3.json
          meta_ridge.joblib
        finger2/ ...
      subject2/ ...

Each finger model is a dict ``{"base_models": (xgb1, xgb2, xgb3), "meta_model": ridge}``.
"""

import joblib
from xgboost import XGBRegressor

from src.config import MODELS_DIR, NUM_FINGERS, NUM_SUBJECTS

NUM_BASE_MODELS = 3


def model_dir(subject, finger, models_dir=MODELS_DIR):
    """Directory for a given 1-indexed subject and finger."""
    return models_dir / f"subject{subject}" / f"finger{finger}"


def save_finger_model(subject, finger, base_models, meta_model, models_dir=MODELS_DIR):
    out = model_dir(subject, finger, models_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i, xgb in enumerate(base_models, start=1):
        xgb.save_model(out / f"xgb{i}.json")
    joblib.dump(meta_model, out / "meta_ridge.joblib")


def load_finger_model(subject, finger, models_dir=MODELS_DIR, device="cpu"):
    d = model_dir(subject, finger, models_dir)
    if not d.exists():
        raise FileNotFoundError(f"No trained model found at {d}")

    base_models = []
    for i in range(1, NUM_BASE_MODELS + 1):
        xgb = XGBRegressor()
        xgb.load_model(d / f"xgb{i}.json")
        # Models may have been trained on GPU; run inference wherever we are.
        xgb.set_params(device=device)
        base_models.append(xgb)

    meta_model = joblib.load(d / "meta_ridge.joblib")
    return {"base_models": tuple(base_models), "meta_model": meta_model}


def load_subject_models(subject, models_dir=MODELS_DIR, device="cpu"):
    """Load all finger models for one subject, ordered finger 1..5."""
    return [load_finger_model(subject, f, models_dir, device)
            for f in range(1, NUM_FINGERS + 1)]


def available_subjects(models_dir=MODELS_DIR):
    """Subjects (1-indexed) that have a complete set of finger models on disk."""
    found = []
    for s in range(1, NUM_SUBJECTS + 1):
        if all((model_dir(s, f, models_dir) / "meta_ridge.joblib").exists()
               for f in range(1, NUM_FINGERS + 1)):
            found.append(s)
    return found


def expected_num_features(subject_models):
    """Number of R-matrix columns the subject's models were trained on."""
    return subject_models[0]["base_models"][0].n_features_in_
