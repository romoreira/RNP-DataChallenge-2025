# TRACE, Traceroute based Internet Route Change Analysis with Ensemble Learning 🌐📈

This repository contains the source code and scripts used in our study, **TRACE, Traceroute based Internet Route change Analysis with Ensemble Learning**.

Also, this repository corresponds to our **CT Mon Data Challenge 2025 winning entry**.

Repository, https://github.com/romoreira/RNP-DataChallenge-2025

## Overview

Detecting Internet routing instability is challenging when relying only on endpoint active measurements. TRACE addresses this by detecting route change events using **only traceroute latency data**, remaining **independent from control plane information**.

The pipeline combines,

- Robust feature engineering that captures statistical, temporal, and aggregate context patterns
- A stacked ensemble of gradient boosted decision trees
- F1 oriented threshold calibration to handle rare events and severe class imbalance

## Data

The traceroute dataset used in this work is drawn from the **Measurement Lab, M Lab** open repository.

Each traceroute instance includes,

- Source and destination identifiers
- RTT vector
- Probe outcome counters
- Binary label `route_changed`

Scale used in the experiments,

- Total instances, 28,521,656
- Train split, 19,965,159 rows, 70 percent
- Test split, 8,556,497 rows, 30 percent
- Average probes per traceroute, 1.44

Class distribution in training,

- Class 0, stable, 19,595,589, 98.15 percent
- Class 1, changed, 369,570, 1.85 percent
- Imbalance ratio, about 1 to 53

## Method, TRACE Pipeline

TRACE is organised into four phases.

### Phase 1, Feature engineering

Each raw traceroute row is converted into a feature vector capturing,

- Per trace statistics, mean, variance, percentiles, IQR, min, max, length
- Reliability features from probe counters, success rate and loss rate
- Path level temporal context for each source destination pair, deltas, ratios, time gaps
- Rolling statistics over short windows per path, for example 3 and 7 observations
- Aggregate source and destination context, counts, distribution summaries, z score style deviations

### Phase 2, Base learners, Level 0

Three gradient boosted tree classifiers are trained on engineered features,

- LightGBM
- CatBoost
- XGBoost

Training uses stratified 5 fold cross validation to produce out of fold probability predictions for stacking.

### Phase 3, Stacked ensemble, Level 1

The meta feature vector combines,

- Out of fold probabilities from the three base models
- Summary statistics over those probabilities, mean, standard deviation, median, pairwise differences
- Selected original engineered features
- Simple interaction terms and squared prediction terms

The meta model is a **LightGBM** classifier, tuned with **Hyperopt TPE**, using F1 as the optimisation target.

### Phase 4, Baseline models, no stacking

Baselines are trained directly on engineered features to isolate stacking gains,

- Logistic Regression
- Random Forest
- k Nearest Neighbours

## Imbalance handling, Threshold calibration

Because route change events are rare, the default 0.5 threshold is not appropriate.

TRACE performs threshold calibration by scanning candidate thresholds and selecting the value that **maximises F1**, then applying the same procedure consistently across the stacked model and the baselines.

## Results snapshot

On the reported evaluation, TRACE achieved,

- F1 score, 0.869
- Accuracy, 0.895

and outperformed traditional baselines and individual boosted models under severe class imbalance.

## Repository structure, recommended

A practical structure that matches the pipeline is,

    .
    ├── src/
    │   ├── features/            Feature engineering, Phase 1
    │   ├── models/              Base learners and baselines, Phases 2 and 4
    │   ├── stacking/            Meta features and meta model, Phase 3
    │   └── utils/               I O, metrics, splits, logging
    ├── scripts/                 Runnable entry points
    ├── figs/                    Figures used in the paper
    ├── requirements.txt
    └── README.md

## Environment

Recommended,

- Ubuntu 20.04 or later
- Python 3.9 or later

Main dependencies,

- numpy, pandas
- scikit learn
- lightgbm, xgboost, catboost
- hyperopt
- matplotlib, tqdm, joblib

Install,

    python -m venv .venv
    . .venv/bin/activate
    pip install -r requirements.txt

## How to run, suggested workflow

The commands below assume you provide the paths and adapt script names to your repository.

1, Build features, Phase 1

    python scripts/build_features.py \
      --input /path/to/raw_traceroute_data \
      --output data/features.parquet

2, Train base learners with out of fold predictions, Phase 2

    python scripts/train_base_learners_oof.py \
      --features data/features.parquet \
      --folds 5 \
      --outdir outputs/base

3, Build meta features and train the meta model with Hyperopt TPE, Phase 3

    python scripts/train_meta_model.py \
      --features data/features.parquet \
      --oof outputs/base/oof_predictions.parquet \
      --folds 5 \
      --hyperopt_evals 50 \
      --outdir outputs/stacking

4, Calibrate threshold to maximise F1

    python scripts/calibrate_threshold.py \
      --train_probs outputs/stacking/train_probs.parquet \
      --train_labels /path/to/train_labels \
      --metric f1 \
      --out outputs/stacking/threshold.json

5, Evaluate on the test split

    python scripts/evaluate.py \
      --test_probs outputs/stacking/test_probs.parquet \
      --threshold outputs/stacking/threshold.json \
      --out outputs/metrics.json

## Reproducibility checklist ✅

To reproduce results consistently,

- Fix random seeds across numpy, scikit learn, LightGBM, XGBoost, CatBoost
- Use stratified 5 fold splits for out of fold generation
- Log hyperparameters selected by Hyperopt TPE
- Persist the selected threshold value and report it with metrics

## Citation

If you use this repository, please cite the associated paper,

TRACE, Traceroute based Internet Route change Analysis with Ensemble Learning

Add BibTeX here when the venue details are final.

## Contacts

- Rodrigo Moreira, rodrigo@ufv.br
- Raul Suzuki Borges, raul.borges@ufv.br
- Pedro Henrique A. Damaso de Melo, pedro.henrique.melo@ufv.br
- Larissa F. Rodrigues Moreira, larissa.f.rodrigues@ufv.br
- Flávio de Oliveira Silva, flavio@di.uminho.pt

## Licence

Add your preferred licence, for example MIT, BSD 3 Clause, or Apache 2.0.


## CT-Mon Tests datasheet
* SVDD_TEST8.py: 67.67 F1
* SVDD_TEST9.py: 56.33 F1
* SVDD_TEST10.py: 72.57 F1
* SVDD_TEST11.py: 55.96 F1
* SVDD_TEST12.py: 66.00 F1
* SVDD_TEST13.py: 73:62 F1
* SVDD_TEST14.py: 49.73 F1
* SVDD_TEST15.py: 70:58 F1
* SVDD_TEST16.py: 65.17 F1
* SVDD_TEST17.py: 49.53 F1
* SVDD_TEST18.py: 85.23 F1
* SVDD_TEST18.py: 85.50 F1
* SVDD_TEST20.py: 85.24 F1
* SVDD_TEST21.py: 50.55 F1
* SVDD_TEST22.py: 85.31 F1
* SVDD_TEST23.py: 85.32 F1
* SVDD_TEST24.py: 85.47 F1
* SVDD_TEST25.py: 85.73 F1
* SVDD_TEST25.py: 85.74 F1 (best features)
* SVDD_TEST28.py: 85.83 F1
