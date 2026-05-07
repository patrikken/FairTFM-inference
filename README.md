# FairTFM - Fair Predictions on Tabular Data

This repository provides **inference-only** FairTFM. The training code will follow. 

## Quick Start

```bash
pip install -r requirements.txt
```


### FairTFMClassifier

```python
from fairtfm import FairTFMClassifier, compute_fairness_metrics

# Load checkpoint
classifier = FairTFMClassifier(model="path/to/checkpoint")

# Fit on training data
classifier.fit(X_train, y_train, s_train)

# Predict
predictions = classifier.predict(X_test, s_test)
probabilities = classifier.predict_proba(X_test, s_test)

# Get embeddings
embeddings = classifier.transform(X_test, s_test)

# Fairness metrics (returns dict with performance metrics)
compute_fairness_metrics(X_test, y_test, s_test)
```

### ACSPumsDataset

```python
from fairtfm.datasets import ACSPumsDataset

dataset = ACSPumsDataset(
    acs_task="acs_income",      # Income, employment, mobility, travel_time, public_coverage
    states=["CA"],              # State codes
    sensitive_attr_name="SEX"   # SEX, RAC1P (Race), AGEP (Age)
)
dataset.preprocess()
X_train, X_test, y_train, y_test, s_train, s_test = dataset.get_splits()
```


## Supported Tasks

- `acs_income` - Income prediction
- `acs_employment` - Employment status
- `acs_mobility` - Geographic mobility
- `acs_public_coverage` - Public health insurance
- `acs_travel_time` - Travel time to work

## Sensitive Attributes

- `SEX` - Gender
- `RAC1P` - Race (White/Black)
- `AGEP` - Age (binarized by median)

## Code example
For inference example use the [notebook](notebook.ipynb) or the [inference_example.py](inference_example.py)


# Reproducing the paper main results.

For reproducing the paper's results run:

```python
python paper_results.py --full-eval
```

This will evaluate all the models in [`eval_config/eval_models.csv`](eval_config/eval_models.csv) on all 120 tasks in [`eval_config/fairness_tasks_eval.csv`](eval_config/fairness_tasks_eval.csv).

The supplementary material contains all FairTFM checkpoints across lambda values (0.7, 1.0, 10, 25), which are used to generate the Pareto front. These must be downloaded and placed in the `checkpoints` folder at the repository root.

