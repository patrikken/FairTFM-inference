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

def load_model_from_checkpoint(checkpoint_path: str, device: str = None):
    """
    Load a FairTFM model from a checkpoint file.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file
        device (str): Device to load model on ('cpu', 'cuda', 'mps')
    
    Returns:
        FairTFM: Loaded model in eval mode
    """
    from .model import FairTFM
    
    if device is None:
        device = get_default_device()
    
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
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