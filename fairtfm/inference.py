"""
Inference interface for FairTFM model.
Scikit-learn compatible wrapper for fair model inference with fairness metrics.
"""
import numpy as np
import torch
from typing import Optional, Dict
from sklearn.metrics import accuracy_score, roc_auc_score
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
    equal_opportunity_difference
)
from .utils import load_model_from_checkpoint, get_default_device, get_feature_preprocessor
import torch.nn.functional as F


class FairTFMClassifier:
    """
    Scikit-learn compatible wrapper for FairTFM model inference.
    
    Provides high-level interface for predictions with fairness awareness.
    Computes fairness metrics using fairlearn functions.
    
    Example:
        >>> classifier = FairTFMClassifier(model='path/to/checkpoint.pt')
        >>> classifier.fit(X_train, y_train, s_train)
        >>> predictions = classifier.predict(X_test, s_test)
        >>> metrics = classifier.compute_fairness_metrics(X_test, y_test, s_test)
    """
    
    def __init__(
        self,
        model: Optional[torch.nn.Module] | str = None,
        device: Optional[str] = None,
        num_mem_chunks: int = 8
    ):
        """
        Initialize FairTFMClassifier.
        
        Args:
            model: Path to checkpoint file or pre-initialized torch.nn.Module
            device: Device to use ('cpu', 'cuda', 'mps')
            num_mem_chunks: Number of memory chunks for batch processing
        """ 
        
        if device is None: 
            self.device = get_default_device()
        else:
            self.device = device

        self.num_mem_chunks = num_mem_chunks
        if isinstance(model, str):
            self.model = load_model_from_checkpoint(model, self.device) 
        else:
            self.model = model 
            assert self.model is not None, "Model must be provided either as a torch.nn.Module or a checkpoint file path."
         
        self.X_train = None
        self.y_train = None
        self.s_train = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        s_train: np.ndarray,
    ):
        """
        Fit the classifier on training data.
        
        Args:
            X_train: Training features of shape (n_samples, n_features)
            y_train: Training labels of shape (n_samples,)
            s_train: Sensitive attributes of shape (n_samples,)
        
        Returns:
            self: Fitted classifier
        """ 
        
        # Store training data for inference
        self.feature_preprocessor = get_feature_preprocessor(X_train)
        self.X_train = self.feature_preprocessor.fit_transform(X_train)
        self.y_train = y_train
        self.s_train = s_train
        self.num_classes = max(set(y_train))+1
        
        return self
    def transform(self, X: np.ndarray, s: np.ndarray, beta: float=None) -> np.ndarray:
        """ creates (x,y), runs it through our PyTorch Model and returns the output of the last layer before the softmax, which can be used as a representation of the input data """
        x = np.concatenate((self.X_train, self.feature_preprocessor.transform(X)))
        y = self.y_train
        s = np.concatenate((self.s_train, s), axis=0)
        with torch.no_grad():
            x = torch.from_numpy(x).unsqueeze(0).to(torch.float).to(self.device)  # introduce batch size 1
            y = torch.from_numpy(y).unsqueeze(0).to(torch.float).to(self.device)
            s = torch.from_numpy(s).unsqueeze(0).to(torch.float).to(self.device)
            _, emb = self.model((x, y, s), single_eval_pos=len(self.X_train), num_mem_chunks=self.num_mem_chunks, return_embeddings=True) # remove batch size 1
            emb = emb.squeeze(0)
            emb = emb[len(self.X_train):, -1, :]
            return emb.cpu().numpy()

    def predict(
        self,
        X_test: np.ndarray,
        s_test: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Make predictions on test data.
        Calls predict_proba and picks the class with the highest probability for each datapoint
        Args:
            X_test: Test features of shape (n_test, n_features)
            s_test: Test sensitive attributes of shape (n_test,). If None, uses training sensitive attr mean.
        
        Returns:
            Predicted class labels of shape (n_test,)
        """ 
        predicted_probabilities = self.predict_proba(X_test, s_test)
        return predicted_probabilities.argmax(axis=1)

    def predict_proba(self, X_test: np.ndarray, s_test: np.ndarray) -> np.ndarray:
        """
        creates (x,y), runs it through our PyTorch Model, cuts off the classes that didn't appear in the training data
        and applies softmax to get the probabilities
        """
        x = np.concatenate((self.X_train, self.feature_preprocessor.transform(X_test)))
        y = self.y_train
        #if np.max(s_test) == 0:
        #    s_padded = np.ones_like(s_test)    
        #else: 
        #    s_padded = np.zeros_like(s_test)  # assuming s is a 1D array, adjust if it's not
        s = np.concatenate((self.s_train, s_test), axis=0)
        with torch.no_grad():
            x = torch.from_numpy(x).unsqueeze(0).to(torch.float).to(self.device)  # introduce batch size 1
            y = torch.from_numpy(y).unsqueeze(0).to(torch.float).to(self.device)
            s = torch.from_numpy(s).unsqueeze(0).to(torch.float).to(self.device)
            out = self.model((x, y, s), single_eval_pos=len(self.X_train), num_mem_chunks=self.num_mem_chunks).squeeze(0)  # remove batch size 1
            # our pretrained classifier supports up to num_outputs classes, if the dataset has less we cut off the rest
            out = out[:, :self.num_classes]
            # apply softmax to get a probability distribution
            probabilities = F.softmax(out, dim=1)
            return probabilities.to('cpu').numpy()

    def compute_fairness_metrics(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        s_test: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute fairness metrics using fairlearn functions.
        
        Includes demographic parity, equalized odds, and other fairness measures.
        
        Args:
            X_test: Test features
            y_test: True labels
            s_test: Sensitive attributes
        
        Returns:
            Dictionary with fairness metrics including:
                - accuracy: Overall accuracy
                - auc: Area under the ROC curve (for binary classification)
                - demographic_parity_difference: demographic parity difference
                - equalized_odds_difference: equalized odds difference
                - equal_opportunity_difference: equal opportunity difference
        """

        y_prob = self.predict_proba(X_test, s_test)
        y_pred = y_prob.argmax(axis=1) # convert probabilities to class predictions
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred), 
            'auc': roc_auc_score(y_test, y_prob[:, 1]) if len(np.unique(y_test)) == 2 else None, # only compute AUC for binary classification
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
                pass
        
        return metrics
