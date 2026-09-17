# be521_project_2025

ECoG finger movement decoding project implemented by the team "def calculate feature windows" comprising of Juee Naik, Jadd El Husseini, and Samhitha Vedire. This work was part of the course BE5210 taught by Prof. Brian Litt.
The dataset used for the project is from the 4th International Brain Computer Interfaces Competition, which can be found [here](https://www.bbci.de/competition/iv/).

The original research is in [project_files/5384.ipynb](project_files/5384.ipynb) and the write-up in [BCI_Final_Project_Report.pdf](BCI_Final_Project_Report.pdf). The repo packages that pipeline as a Streamlit web app deployed via Docker on AWS App Runner.

## How it works

```
uploaded .mat ──► src/preprocessing.py ──► src/features.py ──► src/prediction.py ──► trajectory plot
                  CAR, bandpass, notch      windowed features    3×XGBoost + Ridge
                                            + lags + R matrix    per finger, upsample
                                                    ▲
                                              models/subjectN/fingerM/
```

## Layout

```
be521_project_2025/
├── app.py                 Streamlit web app
├── src/
│   ├── config.py          shared constants (fs, window sizes, lags, paths)
│   ├── preprocessing.py   .mat loading, CAR, bandpass + notch filters, LMP, Hilbert band power
│   ├── features.py        per-window features, time lags, R matrix, glove downsampling
│   ├── prediction.py      ensemble inference + Akima upsampling / smoothing
│   └── model_utils.py     save / load models from models/
├── models/                trained models (subjectN/fingerM/) — see models/README.md
├── training/train.py      trains + saves the models; not run by the app
├── project_files/         original notebook
├── requirements.txt
├── Dockerfile
└── BCI_Final_Project_Report.pdf
```

## Run locally

```bash
pip install -r requirements.txt

# Train the models once. Needs raw_training_data.mat 
python training/train.py --data raw_training_data.mat --device cpu   # or --device cuda

streamlit run app.py
```

Upload a `.mat` file containing an ECoG array of shape samples × channels (either a
single array or a cell array of subjects like `leaderboard_data.mat`) pick the
matching subject model, and the app plots the predicted flexion of all five
fingers and offers the predictions as CSV / `.mat`.

## Docker

```bash
docker build -t ecog-decoder .
docker run -p 8080:8080 ecog-decoder
```

The image bundles `models/`, so train (or copy in) the models before building.
For AWS App Runner, deploy the image with port `8080` and health-check path
`/_stcore/health`.
