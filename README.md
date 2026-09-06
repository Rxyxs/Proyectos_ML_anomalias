[ 🇺🇸 English ] | [ 🇨🇱 [Leer en Español](README.es.md) ]

# Bank Anomaly Detection

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-11%20detectors-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-supervised-EB5E28)
![Tests](https://img.shields.io/badge/tests-53%20passing-brightgreen?logo=pytest&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Autoencoder-EE4C2C?logo=pytorch&logoColor=white)
![DuckDB](https://img.shields.io/badge/DuckDB-metrics%20store-FFF000?logo=duckdb&logoColor=black)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

Fraud and anomaly detection system for mobile banking transactions, built on the synthetic **PaySim** dataset ([`ealaxi/paysim1`](https://www.kaggle.com/datasets/ealaxi/paysim1) on Kaggle), which simulates financial transactions based on a month of data from a real mobile money service in Africa.

## Honest note on validation

The numbers in this README **come from an actual run** of the pipeline on the full PaySim dataset (6,362,620 rows downloaded via `kagglehub`), not from estimates: `python -m src.unsupervised.train_unsupervised` for Module 2 and `python -m src.unsupervised.benchmark` for Module 3, plus **53/53 unit tests passing** (`pytest tests/`, on synthetic data, no download needed). Fit and scoring times were measured on that same machine (Windows 10, CPU) and are meant for comparing detectors *against each other*, not as an absolute hardware reference.

Two caveats you need in order to read the metrics correctly:

- **The test set is deliberately enriched.** It holds 50,000 normal transactions plus *all* 8,213 available fraudulent ones — 14.1% fraud, against PaySim's real ~0.13%. That's the only way to have enough anomalies to measure Precision@k stably, but it means these PR-AUC values **do not transfer** to production prevalence: in the real world the same model would be substantially less precise.
- **Module 1 (supervised) was not re-run in this session.** Its metrics aren't reported as numbers here; anyone who clones the repo can generate them with `python -m src.models.train`.

## Goal

Identify fraudulent transactions within a highly imbalanced dataset (the `isFraud` class represents a tiny fraction of all transactions), evaluating different supervised modeling approaches and class-balancing techniques to maximize fraud detection while minimizing false positives.

## Project Architecture

```mermaid
flowchart LR
    A["loader.py<br/>kagglehub, PaySim (6.3M rows)"] --> B[preprocessing.py]
    B --> C[build_features.py]
    C --> D["train.py<br/>LogReg / Random Forest / XGBoost"]
    D --> E[(model.joblib<br/>best PR-AUC)]
    C --> F["train_unsupervised.py<br/>Isolation Forest / LOF / MAD-z / Autoencoder, normal-only"]
    F --> G[(isolation_forest.joblib<br/>+ RobustScaler)]
    F --> H[(metrics.duckdb<br/>PR-AUC / Precision@k per run)]
    C --> I["benchmark.py<br/>11 detectors + 3 ensembles, same split"]
    I --> H
    I --> J[/"ranking + correlation<br/>+ PR curves"/]
```

The project follows a modular architecture that clearly separates data ingestion, preprocessing, feature engineering, and modeling, favoring reproducibility and code testability:

```
bank-anomaly-detection/
├── data/
│   ├── raw/              # Original data downloaded from Kaggle (not tracked)
│   └── processed/        # Transformed data ready for modeling (not tracked)
├── notebooks/
│   ├── 01_eda_paysim.ipynb                     # Exploratory analysis of the PaySim dataset
│   ├── 02_unsupervised_anomaly_detection.ipynb # Module 2: zero-day fraud detection
│   └── 03_benchmark_familias_anomalias.ipynb   # Module 3: detector-family benchmark + ensembles
├── src/
│   ├── data/
│   │   ├── loader.py           # Download (kagglehub) and load the PaySim dataset
│   │   └── preprocessing.py    # Cleaning and transformation of raw data
│   ├── features/
│   │   └── build_features.py   # Feature engineering for the model
│   ├── models/
│   │   ├── train.py            # Training, comparison, and model selection
│   │   ├── visualize.py        # Comparative ROC/PR curves and confusion matrices
│   │   └── predict.py          # Inference on new data
│   ├── unsupervised/            # Modules 2 and 3: unsupervised anomaly detection
│   │   ├── loader.py            # Training data (normal-only) and test data (mixed)
│   │   ├── models.py            # Isolation Forest, LOF, MAD-z statistical baseline
│   │   ├── autoencoder.py       # PyTorch Autoencoder (ReLU/GELU/Swish activations)
│   │   ├── families.py          # Module 3: PCA, GMM, Mahalanobis (MCD), kNN, OC-SVM, HBOS, ECOD
│   │   ├── ensemble.py          # Module 3: score combination (ranks, z, z-max)
│   │   ├── benchmark.py         # Module 3: all 11 families compared on the same split
│   │   ├── metrics_store.py     # Comparative metrics persistence (DuckDB)
│   │   ├── style.py             # Shared figure palette and styling
│   │   └── train_unsupervised.py  # Training, evaluation (Precision@k), and plots
│   └── utils/                  # Shared helper functions
├── tests/                 # Unit tests (pytest): preprocessing, features, MAD baseline,
│                           # autoencoder, metrics store, families, ensembles
├── requirements.txt
├── LICENSE
├── README.md
└── README.es.md
```

Every module under `src/` exposes pure, documented functions meant to be imported both from notebooks (exploration) and scripts (production pipeline), avoiding duplicated logic between the two contexts.

## Dataset

**PaySim** is a mobile financial-transaction simulator based on aggregated data from a real mobile money service provider, extended to include injected fraudulent behavior. It includes transaction types like `CASH-IN`, `CASH-OUT`, `DEBIT`, `PAYMENT`, and `TRANSFER`, along with origin and destination balances before and after each operation.

The `isFraud` target column indicates whether a transaction was fraudulent, while `isFlaggedFraud` marks illegitimate mass transfers flagged by the simulated business rules.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Download and initial load of the dataset:

```bash
python -m src.data.loader
```

This downloads the dataset from Kaggle via `kagglehub` (requires configured Kaggle credentials), copies it to `data/raw/paysim.csv`, and prints a summary of dimensions, first rows, and the percentage distribution of the `isFraud` class.

Model training and comparison:

```bash
python -m src.models.train
```

Runs the full pipeline (load → cleaning → features → split) and trains three candidate models (Logistic Regression, Random Forest, and XGBoost), each with imbalanced-class handling (`class_weight="balanced"` / `scale_pos_weight`). Prints a `classification_report`, confusion matrix, ROC-AUC, and PR-AUC per model, saves the one with the best PR-AUC (the most informative metric for fraud, given the extreme class imbalance) to `data/processed/model.joblib`, and generates comparative ROC/Precision-Recall curves and confusion matrices in `data/processed/figures/`.

Unit tests:

```bash
pytest tests/
```

Unsupervised anomaly detection (Module 2):

```bash
python -m src.unsupervised.train_unsupervised
```

Comparative benchmark of detector families (Module 3):

```bash
python -m src.unsupervised.benchmark
```

Trains all 11 families on the same split and scaling, builds the three ensembles on top of their scores, prints the comparison table with metrics and timings, reports which detector pairs are redundant, and saves the ranking, PR curves, and correlation matrix to `data/processed/figures/`, plus one row per detector in the `benchmark_metrics` table of `data/processed/metrics.duckdb`.

## Module 2: Unknown / zero-day fraud detection (unsupervised)

Module 1 trains on already-labeled fraud, so it can only recognize patterns similar to fraud that already happened before. Module 2 covers the complementary case: a genuinely new ("zero-day") fraud scheme doesn't resemble anything seen during training, and a supervised model has no reason to catch it. The approach here is to learn only the shape of normal behavior and flag anything that deviates from it as anomalous, without using a single fraud label during fitting.

**Data**: instead of downloading `mlg-ulb/creditcardfraud` via `kagglehub` (which would require additional Kaggle credentials not configured in this environment), `src/unsupervised/loader.py` reuses the PaySim data already present in `data/raw/paysim.csv` — the same `clean_data`/`build_features` functions from Module 1 — and splits it into:
- **Train**: a sample of normal transactions (`isFraud == 0`); the model never sees a fraud case while fitting.
- **Test**: a sample of normals + *all* available fraudulent transactions, to have enough real anomalies to measure performance against.

The training sample size is deliberately kept bounded (30k rows): Local Outlier Factor in *novelty* mode needs to build a neighbor index and query it for every prediction, which doesn't scale to the full dataset's 6.3M rows.

**Models** (`src/unsupervised/models.py`, `src/unsupervised/autoencoder.py`):
- **Isolation Forest** (`sklearn.ensemble.IsolationForest`) — isolates points via random partitions; anomalies require fewer partitions to become isolated.
- **Local Outlier Factor** (`sklearn.neighbors.LocalOutlierFactor`, `novelty=True`) — compares a point's local density against its nearest neighbors' density.
- **MAD-z statistical baseline** (`MADBaseline`) — a non-iterative baseline: memorizes the median and Median Absolute Deviation (MAD) per feature at fit time, then scores each row by the maximum robust z-score across its features. Robust to the heavy-tailed distribution of amounts/balances, unlike a plain mean/std z-score.
- **Autoencoder** (`src/unsupervised/autoencoder.py`, PyTorch) — a symmetric fully-connected encoder/decoder with a central bottleneck, trained only on normal transactions to minimize reconstruction MSE; the anomaly score is the reconstruction error itself (`reconstruction_error()`), higher = more anomalous. Trained and compared with three activation functions on the same architecture — **ReLU**, **GELU**, and **Swish** (`nn.SiLU`) — to illustrate their effect on reconstruction quality on this tabular dataset.

All four families expose a homogeneous continuous **Anomaly Score** (higher values = more anomalous), over features scaled with `RobustScaler` (fit on training data only) due to the strong skew of amounts and balances.

**Evaluation** (`src/unsupervised/train_unsupervised.py`): PR-AUC and Precision@k/Recall@k (k = 50, 100, 200 — "of the k most anomalous flagged transactions, how many are real fraud?", the question that matters to an analyst with limited review capacity). Generates `data/processed/figures/unsupervised_scores.png` (Anomaly Score distribution by class, Isolation Forest / LOF / MAD baseline), `data/processed/figures/unsupervised_pr_curve.png` (comparative Precision-Recall curve across all models), and `data/processed/figures/autoencoder_activations.png` (PR-AUC by activation function). Serializes Isolation Forest along with its `RobustScaler` to `data/processed/isolation_forest.joblib`, and persists PR-AUC/Precision@k/Recall@k per model and per run to a local DuckDB file (`data/processed/metrics.duckdb`, via `src/unsupervised/metrics_store.py`) for cross-run comparison.

### Model comparison (Module 2)

| Model | Type | Anomaly score | Notes |
|---|---|---|---|
| Isolation Forest | Ensemble, tree-based | `-score_samples` | Handles non-linear boundaries well, fast to train |
| Local Outlier Factor | Density-based (novelty mode) | `-score_samples` | Sensitive to local density variation |
| MAD-z baseline | Statistical, non-iterative | max robust z-score | No hyperparameters, fast, interpretable per-feature |
| Autoencoder (ReLU / GELU / Swish) | Deep learning (PyTorch) | Reconstruction MSE | Captures non-linear feature interactions; activation choice affects reconstruction quality |

### Measured results (Module 2)

| Model | PR-AUC | Precision@50 | Precision@100 | Precision@200 |
|---|---|---|---|---|
| Local Outlier Factor | **0.802** | 1.000 | 1.000 | 1.000 |
| Autoencoder (ReLU) | 0.581 | 0.960 | 0.970 | 0.975 |
| Autoencoder (Swish) | 0.574 | 0.980 | 0.990 | 0.995 |
| Autoencoder (GELU) | 0.573 | 1.000 | 1.000 | 1.000 |
| Isolation Forest | 0.549 | 0.440 | 0.550 | 0.640 |
| MAD-z baseline | 0.140 | 0.240 | 0.140 | 0.100 |

Local Outlier Factor clearly dominates with this feature set. The autoencoder's three activations land within 0.01 PR-AUC of each other (0.581 / 0.574 / 0.573): on 15-column tabular data, the choice of activation is marginal next to the choice of detector family — a point Module 3 develops.

PR-AUC, Precision@k, and Recall@k for each model/run are persisted to `data/processed/metrics.duckdb` (query it with `duckdb.connect(...)` or `src.unsupervised.metrics_store.load_latest_metrics()`).

![Anomaly score distributions](data/processed/figures/unsupervised_scores.png)
![Precision-Recall curve](data/processed/figures/unsupervised_pr_curve.png)
![Autoencoder activation comparison](data/processed/figures/autoencoder_activations.png)

## Module 3: Detector-family benchmark (unsupervised)

Module 2 covers three approaches plus the autoencoder. Module 3 completes the map: it adds the missing families (`src/unsupervised/families.py`), compares all of them on the same split with the same scaling (`src/unsupervised/benchmark.py`), and tests whether combining them helps at all (`src/unsupervised/ensemble.py`).

The point isn't to pile up models. Without labels you can't pick the best detector *before* deploying it, so the actionable questions are two others: **which families are genuinely complementary** (if two rank transactions almost identically, keeping both adds nothing) and **what each point of PR-AUC costs** (in production scoring runs per transaction and fitting runs once a day, so a detector that's slow to fit but fast to score is perfectly viable — and the reverse is not).

### The seven families added

| Detector | Family | What it detects well |
|---|---|---|
| **PCA (reconstruction)** | Linear reconstruction | Rows outside the principal subspace — the autoencoder's **ablation** |
| **Gaussian Mixture** | Parametric density | Low-probability gaps between modes of the distribution |
| **Robust Mahalanobis (MCD)** | Robust covariance | Broken correlations between features, invisible column by column |
| **kNN (k-th distance)** | Global distance | Points far from any dense neighborhood |
| **One-Class SVM (Nyström)** | Kernel boundary | Rows outside the learned envelope of normal behavior |
| **HBOS** | Per-feature histograms | Rare values in individual columns; score decomposes per feature |
| **ECOD** | Empirical CDF tails | Extreme tails, without a single hyperparameter to tune |

`HBOS` and `ECOD` are implemented from scratch (linear cost, no extra dependency); the rest build on scikit-learn. The exact `OneClassSVM` is O(n²)–O(n³) and doesn't finish on a 30k-row set, so this uses the Nyström kernel approximation + `SGDOneClassSVM`, which solves the same problem in linear time.

The three **ensembles** address the fact that these scores live on incompatible scales (a Mahalanobis distance can't be averaged with a log-likelihood): rank average, standardized-score average, and standardized-score maximum.

### Results

| Detector | Family | PR-AUC | ROC-AUC | Precision@100 | fit (s) | score (s) |
|---|---|---|---|---|---|---|
| Gaussian Mixture | Parametric density | **0.807** | 0.948 | 0.99 | 5.05 | 0.13 |
| Local Outlier Factor | Local density | 0.802 | 0.932 | **1.00** | 0.62 | 1.09 |
| Robust Mahalanobis (MCD) | Robust covariance | 0.698 | 0.896 | 0.94 | 1.25 | 0.01 |
| Ensemble — rank average | Ensemble | 0.643 | 0.884 | **1.00** | — | 0.02 |
| Autoencoder (ReLU) | Non-linear reconstruction | 0.581 | 0.850 | 0.97 | 9.61 | 0.01 |
| kNN (k-th distance) | Global distance | 0.552 | 0.853 | 0.85 | 0.10 | 1.09 |
| Isolation Forest | Isolation | 0.549 | 0.850 | 0.55 | 0.51 | 0.37 |
| One-Class SVM (Nyström) | Kernel boundary | 0.494 | 0.830 | 0.86 | 0.33 | 0.40 |
| PCA (reconstruction) | Linear reconstruction | 0.487 | 0.822 | 0.76 | 0.00 | 0.01 |
| Ensemble — z average | Ensemble | 0.486 | 0.833 | 0.96 | — | 0.02 |
| HBOS | Per-feature statistical | 0.480 | 0.800 | 0.59 | 0.01 | 0.02 |
| Ensemble — z max | Ensemble | 0.457 | 0.837 | 0.73 | — | 0.02 |
| ECOD | Empirical CDF tails | 0.273 | 0.672 | 0.52 | 0.02 | 0.11 |
| MAD-z (baseline) | Per-feature statistical | 0.140 | 0.383 | 0.14 | 0.01 | 0.01 |

![Ranking by family](data/processed/figures/benchmark_ranking.png)

**What the numbers say:**

- **Gaussian Mixture (0.807) and LOF (0.802) tie at the top, but for different reasons.** Their Spearman correlation is only 0.41, and their PR curves cross: LOF dominates between recall 0.4 and 0.8, GMM overtakes it above 0.85. Which one you want depends on the team's review capacity, not on aggregate PR-AUC.
- **The linear ablation justifies the autoencoder, but only just.** The autoencoder (0.581) beats PCA (0.487) — the non-linearity is worth a real ~0.09 of PR-AUC. What those 0.09 cost: 9.6 s of fitting against 0.002 s, plus a PyTorch dependency. Neither one comes close to GMM.
- **The MAD-z baseline isn't just weak, it's anti-informative** (ROC-AUC 0.383, below the 0.5 of random guessing). Looking at each column separately fails here because PaySim fraud is a full drain of the origin balance: every individual value stays inside the observed range, while the heavy tails of legitimate transactions *do* produce extreme z-scores. That is exactly the argument for multivariate methods — and the reason to include Mahalanobis (0.698), which sees the same information but through the full covariance.
- **The ensembles don't win, and that is also a result.** The best of them (rank average, 0.643) lands below GMM and LOF: averaging 11 detectors when 8 are mediocre drags the two good ones down. An ensemble helps when its members are of comparable quality, not when best and worst differ by 0.67 of PR-AUC. With one operationally relevant exception: **Precision@100 = 1.00** for the rank average — at the head of the ranking the consensus *is* perfect, and that head is exactly where an analyst looks.

### Which detectors are redundant?

![Correlation between detectors](data/processed/figures/benchmark_correlation.png)

Spearman correlation between the anomaly *rankings*. **No pair exceeds 0.9**: the eleven families order transactions differently, so none can be dropped for pure redundancy. The closest pairs are Mahalanobis ↔ kNN (0.88) and Mahalanobis ↔ Autoencoder (0.87); the most complementary, MAD-z ↔ LOF (−0.04) and MAD-z ↔ GMM (−0.42).

![Precision-Recall curves](data/processed/figures/benchmark_pr_curve.png)

### Computational cost

The linear-cost detectors (HBOS, ECOD, MAD-z, PCA) score all 58,213 test rows in hundredths of a second and hold nothing in memory beyond a few per-feature vectors. The distance-based ones (LOF, kNN) pay ~1.1 s because every prediction queries a neighbor index against the 30,000 training rows — the factor that decides whether they're viable in an online transaction flow. The autoencoder inverts the relationship: slowest to fit (9.6 s) and among the fastest to score (0.01 s), which is the right profile for production.

## Tech Stack

- **pandas / numpy** — data manipulation and analysis
- **scikit-learn** — preprocessing pipelines and base models
- **xgboost** — gradient-boosting model for fraud classification
- **imbalanced-learn** — resampling techniques (SMOTE, undersampling) for class imbalance
- **matplotlib / seaborn** — exploratory visualization
- **pytorch** — Autoencoder for anomaly detection via reconstruction error
- **scipy** — Spearman correlation and ranking for the detector ensembles
- **duckdb** — local persistence of comparative metrics across runs
- **pytest** — unit tests
- **kagglehub** — programmatic dataset download from Kaggle

## License

MIT — see [LICENSE](LICENSE).

## Author

**Pablo Reyes** — [github.com/Rxyxs](https://github.com/Rxyxs)
