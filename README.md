# FairTFM - Pretrained Tabular Foundation Model for fair predictions on Tabular Data

This repository provides **inference-only** FairTFM. The training code will follow. 

## Quick Start

```bash
pip install -r requirements.txt
```

### Model checkpoints

By default, `FairTFMClassifier()` automatically downloads the default checkpoint (λ = 0.7) from the [Hugging Face Hub](https://huggingface.co/patrikken/FairTFM) — no manual download needed:

```python
from fairtfm import FairTFMClassifier

classifier = FairTFMClassifier()  # downloads FairTFM-0.7-epoch_10000.pt from Hugging Face
```

You can also point it at a specific checkpoint, either a local file, a Hugging Face repo id, or a full Hugging Face URL:

```python
classifier = FairTFMClassifier(model="patrikken/FairTFM")  # repo id, uses default filename
classifier = FairTFMClassifier(model="https://huggingface.co/patrikken/FairTFM/blob/main/FairTFM-25-epoch_10000.pt")
classifier = FairTFMClassifier(model="path/to/local/checkpoint.pt")
```

All checkpoints produced during training across lambda values (0.7, 1.0, 10, 25), which are used to generate the fairness/accuracy Pareto front, are also bundled and available for downloading from [this](https://drive.google.com/uc?export=download&id=1SBztiK9SZZ_6-3I8oT8KadOR0JmuN-j7) google drive link. Higher λ trades predictive performance for lower fairness-metric disparity, so depending on your use case a different checkpoint may give a stronger fairness/accuracy tradeoff — download the bundle and select the checkpoint that fits your needs:

```sh
pip install gdown
gdown 1SBztiK9SZZ_6-3I8oT8KadOR0JmuN-j7

```

or use curl 

```
wget --no-check-certificate "https://drive.google.com/uc?export=download&id=1SBztiK9SZZ_6-3I8oT8KadOR0JmuN-j7" -o checkpoints.zip
```

### FairTFMClassifier interface overview

```python
from fairtfm import FairTFMClassifier, compute_fairness_metrics

# Load the default checkpoint from the Hugging Face Hub
classifier = FairTFMClassifier()

# or load a specific checkpoint (local path, Hugging Face repo id/URL)
classifier = FairTFMClassifier(model="path/to/checkpoint")

# Fit on training data
classifier.fit(X_train, y_train, s_train)

# Predict
predictions = classifier.predict(X_test, s_test)
probabilities = classifier.predict_proba(X_test, s_test)

# Get embeddings of the testing data
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


