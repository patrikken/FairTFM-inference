"""
Utility functions for FairTFM model inference.
"""
import pandas as pd
from typing import Dict, Literal 
import torch
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer
from sklearn.metrics import accuracy_score, roc_auc_score
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
    equal_opportunity_difference,
)


def get_default_device() -> Literal['cpu', 'mps', 'cuda']:
    """Detect and return the best available device for computation."""
    device = 'cpu'
    if torch.backends.mps.is_available():
        device = 'mps'
    if torch.cuda.is_available():
        device = 'cuda'
    return device


def set_randomness_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def to_pandas(x):
    return pd.DataFrame(x) if not isinstance(x, pd.DataFrame) else x

def to_numeric(x):
    return x.apply(pd.to_numeric, errors='coerce').to_numpy()

_DEFAULT_HF_REPO_ID = "patrikken/FairTFM"
_DEFAULT_HF_FILENAME = "FairTFM-0.7-epoch_10000.pt"


def _parse_hf_url(url: str):
    """Parse a huggingface.co model URL into (repo_id, filename, revision)."""
    from urllib.parse import urlparse

    parts = urlparse(url).path.strip('/').split('/')
    # e.g. patrikken/FairTFM/blob/main/FairTFM-0.7-epoch_10000.pt
    repo_id = '/'.join(parts[:2])
    revision = parts[3] if len(parts) > 3 else 'main'
    filename = '/'.join(parts[4:]) if len(parts) > 4 else _DEFAULT_HF_FILENAME
    return repo_id, filename, revision


def load_model_from_checkpoint(
    checkpoint_path: str = None,
    device: str = None,
    filename: str = _DEFAULT_HF_FILENAME,
    revision: str = None,
    cache_dir: str = None,
    force_download: bool = False,
):
    """
    Load a FairTFM model from a local checkpoint file or from the Hugging Face Hub.

    Args:
        checkpoint_path (str): One of:
            - a path to a local checkpoint file on disk
            - a Hugging Face Hub repo id (e.g. "patrikken/FairTFM")
            - a full Hugging Face model URL
              (e.g. "https://huggingface.co/patrikken/FairTFM/blob/main/FairTFM-0.7-epoch_10000.pt")
            - None (default), which downloads the default checkpoint from
              "patrikken/FairTFM" on the Hugging Face Hub
        device (str): Device to load model on ('cpu', 'cuda', 'mps')
        filename (str): Checkpoint filename to fetch from the Hub, used when
            checkpoint_path is a repo id or None. Ignored for local paths and
            full URLs (the filename is parsed from the URL).
        revision (str): Optional Hub revision (branch, tag, or commit hash).
        cache_dir (str): Optional custom Hugging Face cache directory.
        force_download (bool): Force re-download even if a cached copy exists.

    Returns:
        FairTFM: Loaded model in eval mode
    """
    from .model import FairTFM
    import os

    if device is None:
        device = get_default_device()

    resolved_path = checkpoint_path
    if checkpoint_path is None or not os.path.isfile(checkpoint_path):
        from huggingface_hub import hf_hub_download

        repo_id = checkpoint_path or _DEFAULT_HF_REPO_ID
        hub_filename = filename
        hub_revision = revision
        if repo_id.startswith('http://') or repo_id.startswith('https://'):
            repo_id, hub_filename, hub_revision = _parse_hf_url(repo_id)
            hub_revision = revision or hub_revision

        resolved_path = hf_hub_download(
            repo_id=repo_id,
            filename=hub_filename,
            revision=hub_revision,
            cache_dir=cache_dir,
            force_download=force_download,
        )

    state_dict = torch.load(resolved_path, map_location=device, weights_only=False)
    
    model = FairTFM(
        embedding_size=state_dict['architecture']['embedding_size'],
        num_attention_heads=state_dict['architecture']['num_attention_heads'],
        mlp_hidden_size=state_dict['architecture']['mlp_hidden_size'],
        num_layers=state_dict['architecture']['num_layers'],
        num_outputs=state_dict['architecture']['num_outputs'],
        sensitive_attr_hidden_size=state_dict['architecture'].get('sensitive_attr_hidden_size', 96)
    )
    
    model.load_state_dict(state_dict['model'])
    model.to(device)
    model.eval()
    
    return model


def get_feature_preprocessor(X: np.ndarray | pd.DataFrame) -> ColumnTransformer:
    """
    fits a preprocessor that imputes NaNs, encodes categorical features and removes constant features
    """
    X = pd.DataFrame(X)
    num_mask = []
    cat_mask = []
    for col in X:
        unique_non_nan_entries = X[col].dropna().unique()
        if len(unique_non_nan_entries) <= 1:
            num_mask.append(False)
            cat_mask.append(False)
            continue
        non_nan_entries = X[col].notna().sum()
        numeric_entries = pd.to_numeric(X[col], errors='coerce').notna().sum() # in case numeric columns are stored as strings
        num_mask.append(non_nan_entries == numeric_entries)
        cat_mask.append(non_nan_entries != numeric_entries)
        # num_mask.append(is_numeric_dtype(X[col]))  # Assumes pandas dtype is correct

    num_mask = np.array(num_mask)
    cat_mask = np.array(cat_mask)

    num_transformer = Pipeline([
        ("to_pandas", FunctionTransformer(to_pandas)), # to apply pd.to_numeric of pandas
        ("to_numeric", FunctionTransformer(to_numeric)), # in case numeric columns are stored as strings
        ('imputer', SimpleImputer(strategy='mean', add_indicator=True)) # median might be better because of outliers
    ])
    cat_transformer = Pipeline([
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)),
        ('imputer', SimpleImputer(strategy='most_frequent', add_indicator=True)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_mask),
            ('cat', cat_transformer, cat_mask)
        ]
    )
    return preprocessor


def compute_fairness_metrics(
        y_prob: np.ndarray, 
        y_test: np.ndarray,
        s_test: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute fairness metrics using fairlearn functions.
        
        Includes demographic parity, equalized odds, and other fairness measures.
        
        Args:
            y_prob: Predicted probabilities
            X_test: Test features
            y_test: True labels
            s_test: Sensitive attributes
        
        Returns:
            Dictionary with fairness metrics including:
                - accuracy: Overall accuracy
                - balanced_accuracy: Balanced accuracy across classes
                - group_X_accuracy: Accuracy for each sensitive group X
                - demographic_parity_difference: Fairlearn demographic parity difference
                - equalized_odds_difference: Fairlearn equalized odds difference
                - tpr_difference: Difference in true positive rates across groups
                - fpr_difference: Difference in false positive rates across groups
                - accuracy_gap: Max - min accuracy across groups
        """
        
        y_pred = y_prob.argmax(axis=1) # convert probabilities to class predictions
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred), 
            "auc": roc_auc_score(y_test, y_prob[:, 1]) if len(np.unique(y_test)) == 2 else None, # only compute AUC for binary classification
        }
         
         
        
        # Fairlearn metrics for binary classification
        if len(np.unique(y_test)) == 2:
            try:
                # Demographic parity difference
                metrics['demographic_parity_difference'] = demographic_parity_difference(
                    y_true=y_test,
                    y_pred=y_pred,
                    sensitive_features=s_test
                )
                
                # Equalized odds difference
                metrics['equalized_odds_difference'] = equalized_odds_difference(
                    y_true=y_test,
                    y_pred=y_pred,
                    sensitive_features=s_test
                )

                metrics['equal_opportunity_difference'] = equal_opportunity_difference(
                        y_true=y_test,
                        y_pred=y_pred,
                        sensitive_features=s_test
                    )
                    
            except Exception as e:
                # Fallback if fairlearn has issues
                print(f"Warning: Fairlearn metrics could not be computed due to error: {e}")
                pass
        
        return metrics