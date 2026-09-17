# Trained models

Populated by `python training/train.py`. Layout:

```
models/
├── subject1/
│   ├── finger1/
│   │   ├── xgb1.json          # XGBRegressor (max_depth=4, lr=0.03)
│   │   ├── xgb2.json          # XGBRegressor (max_depth=5, lr=0.02, colsample=0.85)
│   │   ├── xgb3.json          # XGBRegressor (max_depth=5, lr=0.02)
│   │   └── meta_ridge.joblib  # RidgeCV stacking the three base models
│   ├── finger2/
│   └── ... finger5/
├── subject2/
└── subject3/
```

Models are subject-specific: the feature dimension depends on the channel
count of that subject's recording (62 / 48 / 64 channels), so a Subject 1
model cannot decode a Subject 2 recording.

`src/model_utils.py` handles reading and writing this layout.
