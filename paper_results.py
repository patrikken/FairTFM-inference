"""
Model Evaluation and Pareto Front Analysis for Fair ML

This script performs comprehensive evaluation of models from eval_config/eval_models.csv:
1. Loads model configurations from eval_config/eval_models.csv
2. Supports multiple model types: TFMs (TabICL, TabPFN, and FairTFM) and sklearn
3. Evaluates each model on all fairness datasets using:
   - Accuracy and AUC
   - Demographic Parity Difference
   - Equalized Odds Difference
4. Constructs Pareto-optimal frontiers per dataset and overall
5. Generates Pareto plots

Author: Generated for Fair-PFN Project
"""

import os
import json
import glob
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import accuracy_score, roc_auc_score
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference, equal_opportunity_difference
from sklearn.neighbors import KNeighborsClassifier
from fairtfm import FairTFMClassifier, ACSPumsDataset, get_default_device, set_randomness_seed
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
#from tabpfn import TabPFNClassifier
#from tabicl import TabICLClassifier


# ============================================================================
# CONFIGURATION & DATA STRUCTURES
# ============================================================================

@dataclass
class EvaluationMetrics:
    """Container for all computed metrics for a checkpoint–dataset pair."""
    checkpoint_name: str
    dataset_name: str
    accuracy: float
    auc: float
    dp_diff: float = 0.0
    eod_diff: float = 0.0
    eop_diff: float = 0.0  # Equal Opportunity Difference
    base_model_name: Optional[str] = None  # Base model name for grouping checkpoints by color
    model_type: Optional[str] = None  # Model type (fair, nanopfn, sklearn) for grouping by model_type
    
    def __post_init__(self):
        """Ensure all metrics are floats."""
        self.accuracy = float(self.accuracy)
        self.auc = float(self.auc)
        self.dp_diff = float(self.dp_diff)
        self.eod_diff = float(self.eod_diff)
        self.eop_diff = float(self.eop_diff)
        # Default base_model_name to checkpoint_name if not set
        if self.base_model_name is None:
            self.base_model_name = self.checkpoint_name


@dataclass
class ParetoPoint:
    """Represents a point in Pareto front analysis."""
    checkpoint_name: str
    metric_x: float  # e.g., accuracy
    metric_y: float  # e.g., dp_diff (lower is better)
    dataset_name: Optional[str] = None
    is_dominated: bool = False
    base_model_name: Optional[str] = None  # Base model for grouping checkpoints by color
    model_type: Optional[str] = None  # Model type for grouping by model_type
    
    def __post_init__(self):
        """Default base_model_name to checkpoint_name if not set."""
        if self.base_model_name is None:
            self.base_model_name = self.checkpoint_name


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across all datasets for a single checkpoint."""
    checkpoint_name: str
    accuracy_mean: float
    auc_mean: float
    dp_diff_mean: float
    eod_diff_mean: float
    eop_diff_mean: float  # Equal Opportunity Difference
    accuracy_std: float = 0.0
    auc_std: float = 0.0
    dp_diff_std: float = 0.0
    eod_diff_std: float = 0.0
    eop_diff_std: float = 0.0
    base_model_name: Optional[str] = None
    model_type: Optional[str] = None


class ModelManager:
    """Manages loading of models from eval_models.csv and discovers all checkpoints."""
    
    def __init__(self, eval_models_csv: str = "eval_config/eval_models_latest.csv"):
        """
        Initialize model manager from CSV configuration.
        
        Args:
            eval_models_csv: Path to eval_models.csv with model configurations
        """
        self.eval_models_csv = eval_models_csv
        self.models = self._load_model_configs()
        
    def _load_model_configs(self) -> List[Tuple[str, str, str, str]]:
        """
        Load model configurations from CSV and discover all checkpoints in each directory.
        
        Returns:
            List of (base_model_name, checkpoint_name, model_type, checkpoint_path) tuples
        """
        models_df = pd.read_csv(self.eval_models_csv)
        models = []
        
        for _, row in models_df.iterrows():
            model_name = row["model_name"]
            model_type = row["model_type"]
            model_path = row["model_path"]
            
            # Skip sklearn models - they don't have checkpoints
            if model_type == "sklearn":
                models.append((model_name, model_name, model_type, model_path))
                continue
            
            # For deep learning models, discover all checkpoints in the directory
            checkpoint_path = Path(model_path)
            checkpoint_dir = checkpoint_path.parent
            
            if not checkpoint_dir.exists():
                print(f"⚠ Model directory not found: {checkpoint_dir} (skipping {model_name})")
                continue
            
            # Find all checkpoint files (.pth and .pt files) in the directory
            checkpoint_files = sorted(list(checkpoint_dir.glob("*.pth")) + list(checkpoint_dir.glob("*.pt")))
            
            if not checkpoint_files:
                print(f"⚠ No checkpoint files found in: {checkpoint_dir} (skipping {model_name})")
                continue
            
            # Sort to process in order: latest_checkpoint first, then others
            latest_first = sorted(checkpoint_files, 
                                 key=lambda f: (f.name != "latest_checkpoint.pth", f.name))
            
            for ckpt_file in latest_first:
                # Create readable checkpoint identifier
                if ckpt_file.name == "latest_checkpoint.pth":
                    ckpt_id = f"{model_name}"
                else:
                    ckpt_id = f"{model_name}_{ckpt_file.stem}"
                
                models.append((model_name, ckpt_id, model_type, str(ckpt_file)))
                print(f"  Found checkpoint: {ckpt_id} → {ckpt_file.name}")
        
        print(f"\nLoaded {len(models)} total checkpoints from CSV")
        return models
    
    def load_model(self, model_type: str, model_path: str):
        """
        Load and instantiate a model based on its type.
        
        Args:
            model_type: Type of model (fair, nanopfn, sklearn)
            model_path: Path to checkpoint or name of sklearn model
            
        Returns:
            Instantiated classifier ready for evaluation
        """
        device = get_default_device()
        
        if model_type == "fair":
            return FairTFMClassifier(model=model_path, device=device) 
        elif model_type == "sklearn":
            if model_path == "LR":
                return LogisticRegression(max_iter=1000)
            elif model_path == "RF":
                return RandomForestClassifier(n_estimators=100)
            elif model_path == "XGB":
                return XGBClassifier(eval_metric='logloss')
            elif model_path == "TabICL":
                return TabICLClassifier()
            elif model_path == "TabPFN":
                return TabPFNClassifier()
            elif model_path == "KNN":
                return KNeighborsClassifier(n_neighbors=5)
            else:
                raise ValueError(f"Unknown sklearn model: {model_path}")
        else:
            raise ValueError(f"Unknown model type: {model_type}")


class DatasetManager:
    """Manages loading and preprocessing of fairness datasets."""
    
    def __init__(self, max_samples: int = 25000):
        """
        Initialize dataset manager.
        
        Args:
            max_samples: Maximum samples per dataset for computational efficiency
        """
        self.max_samples = max_samples
        self.datasets = self._load_datasets()
        
    def _load_datasets(self) -> List[Tuple[str, tuple]]:
        """
        Load all fairness datasets from eval_config/fairness_tasks_eval.csv
        
        Returns:
            List of (dataset_name, data_tuple) where data_tuple = 
            (X_train, X_test, y_train, y_test, s_train, s_test)
        """
        datasets = []
        dataset_df = pd.read_csv("eval_config/fairness_tasks_eval.csv")
        
        for _, row in dataset_df.iterrows():
            task_name = row["task_name"]
            sensitive_attr = row["sensitive_attribute"]
            
            try:
                task = ACSPumsDataset(
                        acs_task=task_name,
                        states=None if row["states"] == "all_states" else eval(row["states"]),
                        survey_year=row["survey_year"],
                        horizon=row["horizon"],
                        sensitive_attr_name=sensitive_attr,
                        max_samples=self.max_samples,
                        test_size=.3
                    ) 
                
                task.preprocess()
                X_train, X_test, y_train, y_test, s_train, s_test = task.get_splits()
                
                dataset_key = task.name
                datasets.append((dataset_key, (X_train, X_test, y_train, y_test, s_train, s_test)))
                
            except Exception as e:
                print(f"⚠ Failed to load dataset {task_name} ({sensitive_attr}): {e}")
                continue
        
        print(f"Loaded {len(datasets)} fairness datasets")
        return datasets


class MetricsComputer:
    """Computes fairness and accuracy metrics."""
    
    @staticmethod
    def compute_metrics(
        classifier,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        s_train: np.ndarray,
        s_test: np.ndarray,
        model_type: str = "fair", 
    ) -> Dict[str, float]:
        """
        Compute all metrics for a model–dataset pair.
        
        Args:
            classifier: Trained classifier
            X_train, X_test: Feature arrays
            y_train, y_test: Target labels
            s_train, s_test: Sensitive attributes
            model_type: Type of model (fair, nanopfn, sklearn)
            beta: Fairness hyperparameter (if applicable)
            
        Returns:
            Dictionary of metrics: {accuracy, auc, dp_diff, eod_diff, eop_diff}
        """
        # Fit classifier based on type
        if model_type == "fair":
            classifier.fit(X_train, y_train, s_train)
            prob = classifier.predict_proba(X_test, s_test)
        else:
            # sklearn and nanopfn models don't use sensitive attributes
            classifier.fit(X_train, y_train)
            prob = classifier.predict_proba(X_test)
        
        # Get predictions
        pred = prob.argmax(axis=1)
        
        # Prepare probabilities for AUC computation
        if prob.shape[1] == 2:
            prob_auc = prob[:, 1]
        else:
            prob_auc = prob[:, :1]
        
        # Compute accuracy metrics
        metrics = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "auc": float(roc_auc_score(y_test, prob_auc, multi_class="ovr")) if prob.shape[1] == 2 else float(roc_auc_score(y_test, prob, multi_class="ovr"))
        }
        
        # Fairness metrics (always computed on s_test for consistency)
        if s_test is not None:
            metrics["dp_diff"] = float(demographic_parity_difference(y_test, pred, sensitive_features=s_test))
            metrics["eod_diff"] = float(equalized_odds_difference(y_test, pred, sensitive_features=s_test))
            metrics["eop_diff"] = float(equal_opportunity_difference(y_test, pred, sensitive_features=s_test))
        else:
            metrics["dp_diff"] = 0.0
            metrics["eod_diff"] = 0.0
            metrics["eop_diff"] = 0.0
        return metrics


class ParetoAnalyzer:
    """Constructs and analyzes Pareto fronts."""
    
    @staticmethod
    def identify_pareto_optimal(points: List[ParetoPoint], maximize_x: bool = True) -> List[ParetoPoint]:
        """
        Identify Pareto-optimal points.
        
        For maximize_x=True: Points are non-dominated if no other point has both
        higher metric_x AND lower metric_y (fairness improves = lower magnitude).
        
        Args:
            points: List of ParetoPoint objects
            maximize_x: If True, higher metric_x is better; if False, lower is better
            
        Returns:
            List of non-dominated (Pareto-optimal) points
        """
        if not points:
            return []
        
        pareto_optimal = []
        
        for i, point in enumerate(points):
            is_dominated = False
            
            for j, other_point in enumerate(points):
                if i == j:
                    continue
                
                # Check if 'other_point' dominates 'point'
                x_better = (other_point.metric_x > point.metric_x) if maximize_x else (other_point.metric_x < point.metric_x)
                y_better = abs(other_point.metric_y) < abs(point.metric_y)  # Lower fairness diff is better
                
                if x_better and y_better:
                    is_dominated = True
                    break
            
            if not is_dominated:
                point.is_dominated = False
                pareto_optimal.append(point)
        
        return pareto_optimal
    
    @staticmethod
    def build_pareto_front_per_dataset(
        results: List[EvaluationMetrics],
        dataset_name: str,
        metric_x: str = "accuracy",
        metric_y: str = "dp_diff"
    ) -> List[ParetoPoint]:
        """
        Build Pareto front for a single dataset.
        
        Args:
            results: List of EvaluationMetrics
            dataset_name: Name of the dataset to filter by
            metric_x: X-axis metric (default: accuracy)
            metric_y: Y-axis metric (default: dp_diff)
            
        Returns:
            List of Pareto-optimal points
        """
        dataset_results = [r for r in results if r.dataset_name == dataset_name]
        
        points = []
        for result in dataset_results:
            point = ParetoPoint(
                checkpoint_name=result.checkpoint_name,
                metric_x=getattr(result, metric_x),
                metric_y=getattr(result, metric_y),
                dataset_name=dataset_name
            )
            points.append(point)
        
        return ParetoAnalyzer.identify_pareto_optimal(points, maximize_x=True)
    
    @staticmethod
    def build_overall_pareto_front(
        aggregated_results: List[AggregatedMetrics],
        metric_x: str = "accuracy_mean",
        metric_y: str = "dp_diff_mean"
    ) -> List[ParetoPoint]:
        """
        Build overall Pareto front by aggregating across datasets.
        
        Args:
            aggregated_results: List of AggregatedMetrics
            metric_x: X-axis metric (default: accuracy_mean)
            metric_y: Y-axis metric (default: dp_diff_mean)
            
        Returns:
            List of Pareto-optimal points
        """
        points = []
        for result in aggregated_results:
            point = ParetoPoint(
                checkpoint_name=result.checkpoint_name,
                metric_x=getattr(result, metric_x),
                metric_y=getattr(result, metric_y),
                dataset_name="Overall"
            )
            points.append(point)
        
        return ParetoAnalyzer.identify_pareto_optimal(points, maximize_x=True)
    
    @staticmethod
    def identify_pareto_optimal_by_model_type(points: List[ParetoPoint]) -> List[ParetoPoint]:
        """
        Identify Pareto-optimal points grouped by model_type.
        - Fair models: single Pareto front across all fair models
        - Other models: independent Pareto fronts per model
        
        Args:
            points: List of ParetoPoint objects (must have model_type set)
            
        Returns:
            List of points with is_dominated flags set appropriately
        """
        if not points:
            return []
        
        # Separate points by model_type
        fair_points = [p for p in points if p.model_type == "fair"]
        other_points = [p for p in points if p.model_type != "fair"]
        
        # Compute Pareto optimal for fair models together
        pareto_fair = []
        if fair_points:
            pareto_fair = ParetoAnalyzer.identify_pareto_optimal(fair_points, maximize_x=True)
            # Mark non-dominated fair points
            pareto_fair_set = set((p.checkpoint_name, p.metric_x, p.metric_y) for p in pareto_fair)
            for p in fair_points:
                if (p.checkpoint_name, p.metric_x, p.metric_y) not in pareto_fair_set:
                    p.is_dominated = True
        
        # Compute Pareto optimal for each non-fair model independently
        pareto_other = []
        if other_points:
            # Group by model
            model_groups = {}
            for p in other_points:
                if p.base_model_name not in model_groups:
                    model_groups[p.base_model_name] = []
                model_groups[p.base_model_name].append(p)
            
            # Compute Pareto for each model group
            for model_name, model_points in model_groups.items():
                pareto_model = ParetoAnalyzer.identify_pareto_optimal(model_points, maximize_x=True)
                pareto_other.extend(pareto_model)
                # Mark non-dominated points for this model
                pareto_model_set = set((p.checkpoint_name, p.metric_x, p.metric_y) for p in pareto_model)
                for p in model_points:
                    if (p.checkpoint_name, p.metric_x, p.metric_y) not in pareto_model_set:
                        p.is_dominated = True
        
        return pareto_fair + pareto_other


class Visualizer:
    """Generates publication-ready Pareto front plots."""
    
    # NeurIPS-quality color palette
    PALETTE = {
        "pareto": "#E74C3C",      # Red for Pareto-optimal
        "dominated": "#95A5A6",   # Gray for dominated
        "benchmark": "#3498DB"    # Blue for baselines
    }
    
    # Distinct colors for methods
    METHOD_COLORS = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
        "#c49c94", "#f7b6d2", "#c7c7c7", "#dbbd22", "#9edae5"
    ]
    
    DPI = 300
    FIGSIZE = (10, 7)
    
    @staticmethod
    def setup_matplotlib():
        """Configure matplotlib for publication-quality plots."""
        plt.style.use("default")  # Use clean default style
        plt.rcParams.update({
            "font.size": 16,
            "font.family": "sans-serif",
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 15,
            "figure.dpi": Visualizer.DPI,
            "savefig.dpi": Visualizer.DPI,
            "lines.linewidth": 2,
            "lines.markersize": 8,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.grid": False,  # We'll add grid manually
        })
    
    @staticmethod
    def plot_pareto_front_3subplots(
        points_dp: List[ParetoPoint],
        points_eod: List[ParetoPoint],
        points_eop: List[ParetoPoint],
        title_prefix: str,
        metric_x_label: str,
        output_path_prefix: str,
        method_colors: Optional[Dict[str, str]] = None,
        pareto_dp: Optional[List[ParetoPoint]] = None,
        pareto_eod: Optional[List[ParetoPoint]] = None,
        pareto_eop: Optional[List[ParetoPoint]] = None
    ):
        """
        Generate a 3-subplot figure with Pareto fronts for three fairness metrics.
        - Column 1: Accuracy/AUC vs DP Difference
        - Column 2: Accuracy/AUC vs EOD Difference
        - Column 3: Accuracy/AUC vs Equal Opportunity Difference
        
        Args:
            points_dp, points_eod, points_eop: Points for each metric
            title_prefix: Prefix for titles (e.g., "Pareto Front: dataset_name")
            metric_x_label: X-axis label (e.g., "Accuracy ↑" or "AUC ↑")
            output_path_prefix: Base path for saving (will add suffixes)
            method_colors: Dict mapping base model names to colors
            pareto_dp, pareto_eod, pareto_eop: Pareto-optimal points for each metric
        """
        if pareto_dp is None:
            pareto_dp = [p for p in points_dp if not p.is_dominated]
        if pareto_eod is None:
            pareto_eod = [p for p in points_eod if not p.is_dominated]
        if pareto_eop is None:
            pareto_eop = [p for p in points_eop if not p.is_dominated]
        
        # Collect all points to determine unique methods
        all_points = points_dp + points_eod + points_eop
        if method_colors is None:
            unique_methods = sorted(set(p.base_model_name for p in all_points))
            method_colors = {method: Visualizer.METHOD_COLORS[i % len(Visualizer.METHOD_COLORS)] 
                            for i, method in enumerate(unique_methods)}
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Define metrics info for each subplot
        subplot_data = [
            (points_dp, pareto_dp, "DP Difference ↓"),
            (points_eod, pareto_eod, "EOD Difference ↓"),
            (points_eop, pareto_eop, "EOP Difference ↓")
        ]
        
        # Plot each subplot
        for ax, (points, pareto_points, y_label) in zip(axes, subplot_data):
            unique_methods = sorted(set(p.base_model_name for p in points))
            pareto_set = set((p.checkpoint_name, p.metric_x, p.metric_y) for p in pareto_points)
            
            # Plot all points grouped by base model
            for method in unique_methods:
                method_points = [p for p in points if p.base_model_name == method]
                
                # Separate Pareto-optimal and dominated points
                pareto_method = [p for p in method_points if (p.checkpoint_name, p.metric_x, p.metric_y) in pareto_set]
                dominated_method = [p for p in method_points if (p.checkpoint_name, p.metric_x, p.metric_y) not in pareto_set]
                
                color = method_colors.get(method, "#000000")
                
                # Plot dominated points (lighter circles)
                if dominated_method:
                    x_dominated = [p.metric_x for p in dominated_method]
                    y_dominated = [p.metric_y for p in dominated_method]
                    ax.scatter(x_dominated, y_dominated, 
                              color=color, 
                              alpha=0.2, s=80, marker="o", 
                              zorder=2, edgecolors=color, linewidth=1)
                
                # Plot Pareto-optimal points (filled circles with edges)
                if pareto_method:
                    x_pareto = [p.metric_x for p in pareto_method]
                    y_pareto = [p.metric_y for p in pareto_method]
                    ax.scatter(x_pareto, y_pareto, 
                              color=color, 
                              alpha=0.9, s=120, marker="o", 
                              edgecolors=color, linewidth=1.5,
                              zorder=3)
            
            # Draw Pareto front lines per model_type
            # For fair models: single line across all fair points
            fair_pareto = [p for p in pareto_points if p.model_type == "fair"]
            if fair_pareto:
                sorted_fair = sorted(fair_pareto, key=lambda p: p.metric_x)
                x_line = [p.metric_x for p in sorted_fair]
                y_line = [p.metric_y for p in sorted_fair]
                ax.plot(x_line, y_line, color="#1f1f1f", 
                       linestyle="-", alpha=0.7, linewidth=2.1, zorder=1)
            
            # For non-fair models: per-model solid lines
            non_fair_points = [p for p in pareto_points if p.model_type != "fair"]
            model_groups = {}
            for p in non_fair_points:
                if p.base_model_name not in model_groups:
                    model_groups[p.base_model_name] = []
                model_groups[p.base_model_name].append(p)
            
            pareto_line_colors = ["#d62728", "#2ca02c", "#9467bd"]  # Red, Green, Purple
            #for (model_name, model_points), line_color in zip(model_groups.items(), pareto_line_colors):
            #    sorted_model = sorted(model_points, key=lambda p: p.metric_x)
            #    x_line = [p.metric_x for p in sorted_model]
            #    y_line = [p.metric_y for p in sorted_model]
            #    ax.plot(x_line, y_line, color=line_color, 
            #          linestyle="-", alpha=0.6, linewidth=1.8, zorder=1)
            
            overall_pareto_points = ParetoAnalyzer.identify_pareto_optimal(pareto_points, maximize_x=True)
            sorted_model = sorted(overall_pareto_points, key=lambda p: p.metric_x)
            x_line = [p.metric_x for p in sorted_model]
            y_line = [p.metric_y for p in sorted_model]
            ax.plot(x_line, y_line, color="#000000", 
                      linestyle="-", alpha=0.6, linewidth=1.8, zorder=1)
            # Formatting
            ax.set_xlabel(metric_x_label, fontsize=18)
            ax.set_ylabel(r"Unfairness $\downarrow$", fontsize=18)
            ax.set_title(y_label.split("↓")[0].strip(), fontsize=18, pad=10)
            
            # Grid styling
            ax.grid(True, alpha=0.4, linestyle="-", linewidth=0.7, color="gray")
            ax.set_axisbelow(True)
            
            # Clean spine styling
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(1)
            ax.spines["bottom"].set_linewidth(1)
        
        # Add a shared legend at the bottom (only once for all subplots)
        unique_methods = sorted(set(p.base_model_name for p in all_points))
        legend_elements = []
        for method in unique_methods:
            color = method_colors.get(method, "#000000")
            from matplotlib.lines import Line2D
            legend_elements.append(Line2D([0], [0], marker="o", color="w", 
                                         markerfacecolor=color, markeredgecolor=color, 
                                         markersize=10, label=method.replace("FTFM", "FairTFM"), linewidth=0))
        
        # Place legend below all subplots
        fig.legend(handles=legend_elements, loc="lower center",  bbox_to_anchor=(0.5, -0.075),
                  framealpha=.8, edgecolor="black", fontsize=18, ncol=5,
                  frameon=True, fancybox=False)
        
        #plt.suptitle(title_prefix, fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout(rect=[0, 0.08, 1, 0.98])
        
        # Save in multiple formats
        os.makedirs(os.path.dirname(output_path_prefix) or ".", exist_ok=True)
        
        # Save as PNG
        png_path = output_path_prefix.replace(".pdf", "") if output_path_prefix.endswith(".pdf") else output_path_prefix
        png_path = f"{png_path}_3subplots.png" if not png_path.endswith(".png") else png_path
        #plt.savefig(png_path, dpi=Visualizer.DPI, bbox_inches="tight", format="png")
        
        # Save as PDF
        pdf_path = png_path.replace(".png", ".pdf")
        plt.savefig(pdf_path, dpi=Visualizer.DPI, bbox_inches="tight", format="pdf")
        
        plt.close()
        
        print(f"✓ Saved 3-subplot plot: {png_path}, {pdf_path}")

    @staticmethod
    def plot_pareto_front(
        points: List[ParetoPoint],
        title: str,
        xlabel: str,
        ylabel: str,
        output_path: str,
        method_colors: Optional[Dict[str, str]] = None,
        pareto_points: Optional[List[ParetoPoint]] = None
    ):
        """
        Generate a Pareto front plot with model_type-aware Pareto lines.
        - Fair models: single Pareto line across all fair checkpoints
        - Other models: independent Pareto lines per model
        Points are colored by base_model_name.
        
        Args:
            points: All evaluation points (all checkpoints)
            title: Plot title
            xlabel: X-axis label with direction (e.g., "Accuracy ↑")
            ylabel: Y-axis label with direction (e.g., "DP Difference ↓")
            output_path: Path to save the plot (supports .png, .pdf)
            method_colors: Dict mapping base model names to colors
            pareto_points: Explicitly provided Pareto-optimal points (optional)
        """
        if pareto_points is None:
            pareto_points = [p for p in points if not p.is_dominated]
        
        if method_colors is None:
            # Generate default colors based on base model names
            unique_methods = sorted(set(p.base_model_name for p in points))
            method_colors = {method: Visualizer.METHOD_COLORS[i % len(Visualizer.METHOD_COLORS)] 
                            for i, method in enumerate(unique_methods)}
        
        fig, ax = plt.subplots(figsize=Visualizer.FIGSIZE)
        
        # Plot all points grouped by base model
        unique_methods = sorted(set(p.base_model_name for p in points))
        pareto_set = set((p.checkpoint_name, p.metric_x, p.metric_y) for p in pareto_points)
        
        # First pass: plot all actual data points
        for method in unique_methods:
            method_points = [p for p in points if p.base_model_name == method]
            
            # Separate Pareto-optimal and dominated points for this method
            pareto_method = [p for p in method_points if (p.checkpoint_name, p.metric_x, p.metric_y) in pareto_set]
            dominated_method = [p for p in method_points if (p.checkpoint_name, p.metric_x, p.metric_y) not in pareto_set]
            
            color = method_colors.get(method, "#000000")
            
            # Plot dominated points for this method (lighter circles)
            if dominated_method:
                x_dominated = [p.metric_x for p in dominated_method]
                y_dominated = [p.metric_y for p in dominated_method]
                ax.scatter(x_dominated, y_dominated, 
                          color=color, 
                          alpha=0.2, s=80, marker="o", 
                          zorder=2, edgecolors=color, linewidth=1)
            
            # Plot Pareto-optimal points for this method (filled circles with edges)
            if pareto_method:
                x_pareto = [p.metric_x for p in pareto_method]
                y_pareto = [p.metric_y for p in pareto_method]
                ax.scatter(x_pareto, y_pareto, 
                          color=color, 
                          alpha=0.9, s=120, marker="o", 
                          edgecolors=color, linewidth=1.5,
                          zorder=3)
        
        # Draw Pareto front lines per model_type (solid lines)
        # For fair models: single line across all fair points
        fair_pareto = [p for p in pareto_points if p.model_type == "fair"]
        if fair_pareto:
            sorted_fair = sorted(fair_pareto, key=lambda p: p.metric_x)
            x_line = [p.metric_x for p in sorted_fair]
            y_line = [p.metric_y for p in sorted_fair]
            ax.plot(x_line, y_line, color="#1f1f1f", 
                   linestyle="-", alpha=0.7, linewidth=2.1, zorder=1)
        
        # For non-fair models: per-model solid lines
        non_fair_points = [p for p in pareto_points if p.model_type != "fair"]
        model_groups = {}
        for p in non_fair_points:
            if p.base_model_name not in model_groups:
                model_groups[p.base_model_name] = []
            model_groups[p.base_model_name].append(p)
        
        pareto_line_colors = ["#d62728", "#2ca02c", "#9467bd"]  # Red, Green, Purple for non-fair
        for (model_name, model_points), line_color in zip(model_groups.items(), pareto_line_colors):
            sorted_model = sorted(model_points, key=lambda p: p.metric_x)
            x_line = [p.metric_x for p in sorted_model]
            y_line = [p.metric_y for p in sorted_model]
            ax.plot(x_line, y_line, color=line_color, 
                   linestyle="-", alpha=0.6, linewidth=1.8, zorder=1)
        
        # Second pass: create consistent legend entries for all methods (circle markers)
        for method in unique_methods:
            color = method_colors.get(method, "#000000")
            # Use circles for legend (represent all points from this method)
            ax.scatter([], [], color=color, alpha=0.9, s=120, marker="o", 
                      edgecolors=color, linewidth=1.5, label=method)
        
        # Formatting
        ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
        #ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
        
        # Legend at bottom in horizontal layout
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), 
                 framealpha=1.0, edgecolor="black", fontsize=12, ncol=5,
                 markerscale=1.2, scatterpoints=1, frameon=True, fancybox=False)
        
        # Grid styling - solid lines, visible but not obtrusive
        ax.grid(True, alpha=0.4, linestyle="-", linewidth=0.7, color="gray")
        ax.set_axisbelow(True)
        
        # Clean spine styling
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1)
        ax.spines["bottom"].set_linewidth(1)
        
        plt.tight_layout()
        
        # Save in multiple formats
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        # Save as PNG
        png_path = output_path.replace(".pdf", ".png") if output_path.endswith(".pdf") else output_path
        plt.savefig(png_path, dpi=Visualizer.DPI, bbox_inches="tight", format="png")
        
        # Save as PDF for vector format
        pdf_path = output_path.replace(".png", ".pdf") if output_path.endswith(".png") else output_path
        plt.savefig(pdf_path, dpi=Visualizer.DPI, bbox_inches="tight", format="pdf")
        
        plt.close()
        
        print(f"✓ Saved plot: {png_path}, {pdf_path}")


class CheckpointAnalysisPipeline:
    """Main orchestrator for checkpoint aggregation and analysis."""
    
    def __init__(
        self,
        eval_models_csv: str = "fairness_datasets/eval_models.csv",
        output_dir: str = "./pareto_analysis",
        seed: int = 42,
        generate_per_dataset_plots: bool = False
    ):
        """
        Initialize the analysis pipeline.
        
        Args:
            eval_models_csv: Path to eval_models.csv configuration file
            output_dir: Directory for output results and plots
            seed: Random seed for reproducibility
            generate_per_dataset_plots: If True, generate per-dataset Pareto plots in addition to overall plots (default: False)
        """
        set_randomness_seed(seed)
        
        self.eval_models_csv = eval_models_csv
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.generate_per_dataset_plots = generate_per_dataset_plots
        
        print("Initializing pipeline...")
        self.model_manager = ModelManager(eval_models_csv)
        self.dataset_manager = DatasetManager(max_samples=10000)
        self.metrics_computer = MetricsComputer()
        self.pareto_analyzer = ParetoAnalyzer()
        
        self.evaluation_results: List[EvaluationMetrics] = []
        self.aggregated_results: List[AggregatedMetrics] = []
        self.model_types: Dict[str, str] = {}  # Store model_type for each model_name
    
    def evaluate_all_checkpoints(self) -> List[EvaluationMetrics]:
        """
        Evaluate all models on all datasets.
        
        Returns:
            List of EvaluationMetrics for all model–dataset pairs
        """
        print("\n" + "="*70)
        print("EVALUATING ALL MODELS ON ALL DATASETS")
        print("="*70)
        
        total_pairs = len(self.model_manager.models) * len(self.dataset_manager.datasets)
        completed = 0
        
        for base_model_name, checkpoint_name, model_type, model_path in self.model_manager.models:
            print(f"\n[Checkpoint {checkpoint_name} ({model_type})]")
            self.model_types[base_model_name] = model_type
            
            try:
                classifier = self.model_manager.load_model(model_type, model_path)
            except Exception as e:
                print(f"  ✗ Failed to load checkpoint: {e}")
                continue
            
            for dataset_name, dataset_tuple in self.dataset_manager.datasets:
                X_train, X_test, y_train, y_test, s_train, s_test = dataset_tuple
                
                try:
                    metrics = self.metrics_computer.compute_metrics(
                        classifier, X_train, X_test, y_train, y_test, s_train, s_test,
                        model_type=model_type
                    )
                    
                    result = EvaluationMetrics(
                        checkpoint_name=checkpoint_name,
                        dataset_name=dataset_name,
                        accuracy=metrics["accuracy"],
                        auc=metrics["auc"],
                        dp_diff=metrics["dp_diff"],
                        eod_diff=metrics["eod_diff"],
                        eop_diff=metrics.get("eop_diff", 0.0),
                        base_model_name=base_model_name,
                        model_type=model_type
                    )
                    self.evaluation_results.append(result)
                    
                    completed += 1
                    print(f"  ✓ {dataset_name:30s} | Acc: {result.accuracy:.4f} | DP: {result.dp_diff:7.4f} | EOD: {result.eod_diff:7.4f} | EOP: {result.eop_diff:7.4f}")
                    
                except Exception as e:
                    print(f"  ✗ Error evaluating {dataset_name}: {e}")
                    completed += 1
        
        print(f"\nCompleted {completed}/{total_pairs} evaluations")
        return self.evaluation_results
    
    def aggregate_results(self) -> List[AggregatedMetrics]:
        """
        Aggregate metrics across datasets for each checkpoint.
        
        Returns:
            List of AggregatedMetrics (one per checkpoint)
        """
        print("\n" + "="*70)
        print("AGGREGATING RESULTS ACROSS DATASETS")
        print("="*70)
        
        checkpoint_names = set(r.checkpoint_name for r in self.evaluation_results)
        
        for ckpt_name in sorted(checkpoint_names):
            ckpt_results = [r for r in self.evaluation_results if r.checkpoint_name == ckpt_name]
            
            # Get base_model_name and model_type from first result (all from same checkpoint should have same values)
            base_model_name = ckpt_results[0].base_model_name if ckpt_results else ckpt_name
            model_type = ckpt_results[0].model_type if ckpt_results else None
            
            accuracy_vals = np.array([r.accuracy for r in ckpt_results])
            auc_vals = np.array([r.auc for r in ckpt_results])
            dp_diff_vals = np.array([r.dp_diff for r in ckpt_results])
            eod_diff_vals = np.array([r.eod_diff for r in ckpt_results])
            eop_diff_vals = np.array([r.eop_diff for r in ckpt_results])
            
            agg = AggregatedMetrics(
                checkpoint_name=ckpt_name,
                accuracy_mean=float(np.mean(accuracy_vals)),
                auc_mean=float(np.mean(auc_vals)),
                dp_diff_mean=float(np.mean(np.abs(dp_diff_vals))),  # Use absolute values
                eod_diff_mean=float(np.mean(np.abs(eod_diff_vals))),
                eop_diff_mean=float(np.mean(np.abs(eop_diff_vals))),
                accuracy_std=float(np.std(accuracy_vals)),
                auc_std=float(np.std(auc_vals)),
                dp_diff_std=float(np.std(np.abs(dp_diff_vals))),
                eod_diff_std=float(np.std(np.abs(eod_diff_vals))),
                eop_diff_std=float(np.std(np.abs(eop_diff_vals))),
                base_model_name=base_model_name,
                model_type=model_type
            )
            self.aggregated_results.append(agg)
            
            print(f"{ckpt_name:40s} | Acc: {agg.accuracy_mean:.4f}±{agg.accuracy_std:.4f} | DP: {agg.dp_diff_mean:.4f}±{agg.dp_diff_std:.4f}")
        
        return self.aggregated_results
    
    def generate_pareto_fronts(self):
        """Generate and save Pareto fronts for all datasets and overall."""
        print("\n" + "="*70)
        print("CONSTRUCTING PARETO FRONTS")
        print("="*70)
        
        dataset_names = set(r.dataset_name for r in self.evaluation_results)
        
        # Per-dataset Pareto fronts
        for dataset_name in sorted(dataset_names):
            print(f"\n[Dataset: {dataset_name}]")
            
            # Accuracy vs DP
            pareto_acc_dp = self.pareto_analyzer.build_pareto_front_per_dataset(
                self.evaluation_results, dataset_name, "accuracy", "dp_diff"
            )
            print(f"  Accuracy vs DP Difference: {len(pareto_acc_dp)} Pareto-optimal points")
            
            # Accuracy vs EOD
            pareto_acc_eod = self.pareto_analyzer.build_pareto_front_per_dataset(
                self.evaluation_results, dataset_name, "accuracy", "eod_diff"
            )
            print(f"  Accuracy vs EOD Difference: {len(pareto_acc_eod)} Pareto-optimal points")
            
            # AUC vs DP
            pareto_auc_dp = self.pareto_analyzer.build_pareto_front_per_dataset(
                self.evaluation_results, dataset_name, "auc", "dp_diff"
            )
            print(f"  AUC vs DP Difference: {len(pareto_auc_dp)} Pareto-optimal points")
            
            # AUC vs EOD
            pareto_auc_eod = self.pareto_analyzer.build_pareto_front_per_dataset(
                self.evaluation_results, dataset_name, "auc", "eod_diff"
            )
            print(f"  AUC vs EOD Difference: {len(pareto_auc_eod)} Pareto-optimal points")
        
        # Overall Pareto fronts
        print(f"\n[Overall (Aggregate)]")
        
        pareto_overall_acc_dp = self.pareto_analyzer.build_overall_pareto_front(
            self.aggregated_results, "accuracy_mean", "dp_diff_mean"
        )
        print(f"  Accuracy vs DP Difference: {len(pareto_overall_acc_dp)} Pareto-optimal points")
        
        pareto_overall_acc_eod = self.pareto_analyzer.build_overall_pareto_front(
            self.aggregated_results, "accuracy_mean", "eod_diff_mean"
        )
        print(f"  Accuracy vs EOD Difference: {len(pareto_overall_acc_eod)} Pareto-optimal points")
        
        pareto_overall_auc_dp = self.pareto_analyzer.build_overall_pareto_front(
            self.aggregated_results, "auc_mean", "dp_diff_mean"
        )
        print(f"  AUC vs DP Difference: {len(pareto_overall_auc_dp)} Pareto-optimal points")
        
        pareto_overall_auc_eod = self.pareto_analyzer.build_overall_pareto_front(
            self.aggregated_results, "auc_mean", "eod_diff_mean"
        )
        print(f"  AUC vs EOD Difference: {len(pareto_overall_auc_eod)} Pareto-optimal points")
    
    def visualize_pareto_fronts(self):
        """Generate and save publication-quality Pareto front plots."""
        print("\n" + "="*70)
        print("GENERATING PUBLICATION-QUALITY VISUALIZATIONS")
        print("="*70)
        
        Visualizer.setup_matplotlib()
        
        # Generate consistent colors for all base models
        all_base_models = sorted(set(r.base_model_name for r in self.evaluation_results))
        method_colors = {model: Visualizer.METHOD_COLORS[i % len(Visualizer.METHOD_COLORS)] 
                        for i, model in enumerate(all_base_models)}
        
        print(f"\nBase Model Color Assignment:")
        for model, color in method_colors.items():
            print(f"  {model:30s} → {color}")
        
        dataset_names = sorted(set(r.dataset_name for r in self.evaluation_results))
        
        # Per-dataset plots (3-subplot format) - only if enabled
        if self.generate_per_dataset_plots:
            print(f"\n[Generating per-dataset plots...]")
            for dataset_name in dataset_names:
                dataset_results = [r for r in self.evaluation_results if r.dataset_name == dataset_name]
                
                # Convert to ParetoPoint objects for visualization (with model_type)
                points_acc_dp = [ParetoPoint(r.checkpoint_name, r.accuracy, r.dp_diff, dataset_name, 
                                             base_model_name=r.base_model_name, model_type=r.model_type) 
                                for r in dataset_results]
                points_acc_eod = [ParetoPoint(r.checkpoint_name, r.accuracy, r.eod_diff, dataset_name,
                                              base_model_name=r.base_model_name, model_type=r.model_type) 
                                 for r in dataset_results]
                points_acc_eop = [ParetoPoint(r.checkpoint_name, r.accuracy, r.eop_diff, dataset_name,
                                              base_model_name=r.base_model_name, model_type=r.model_type) 
                                 for r in dataset_results]
                
                # Identify Pareto-optimal for each
                pareto_acc_dp = self.pareto_analyzer.identify_pareto_optimal_by_model_type(points_acc_dp)
                pareto_acc_eod = self.pareto_analyzer.identify_pareto_optimal_by_model_type(points_acc_eod)
                pareto_acc_eop = self.pareto_analyzer.identify_pareto_optimal_by_model_type(points_acc_eop)
                
                # Plot with 3 subplots (Accuracy vs all fairness metrics)
                dataset_key = dataset_name.replace("/", "_")
                Visualizer.plot_pareto_front_3subplots(
                    points_acc_dp, points_acc_eod, points_acc_eop,
                    f"Pareto Front: {dataset_name} (Accuracy)",
                    r"Accuracy $\uparrow$",
                    str(self.output_dir / f"{dataset_key}_accuracy"),
                    method_colors=method_colors,
                    pareto_dp=pareto_acc_dp,
                    pareto_eod=pareto_acc_eod,
                    pareto_eop=pareto_acc_eop
                )
        else:
            print(f"\n[Skipping per-dataset plots (disabled by default)]")
            print(f"  [Enable with --per-dataset-plots flag]")
        
        # Overall plots (aggregate across datasets) with 3 subplots - ALWAYS GENERATED
        print(f"\n[Generating overall Pareto fronts (aggregated across datasets)...]")
        
        # Overall Accuracy plots
        points_overall_acc_dp = [ParetoPoint(r.checkpoint_name, r.accuracy_mean, r.dp_diff_mean, "Overall",
                                             base_model_name=r.base_model_name, model_type=r.model_type)
                                for r in self.aggregated_results]
        points_overall_acc_eod = [ParetoPoint(r.checkpoint_name, r.accuracy_mean, r.eod_diff_mean, "Overall",
                                              base_model_name=r.base_model_name, model_type=r.model_type)
                                 for r in self.aggregated_results]
        points_overall_acc_eop = [ParetoPoint(r.checkpoint_name, r.accuracy_mean, r.eop_diff_mean, "Overall",
                                              base_model_name=r.base_model_name, model_type=r.model_type)
                                 for r in self.aggregated_results]
        
        pareto_overall_acc_dp = self.pareto_analyzer.identify_pareto_optimal_by_model_type(points_overall_acc_dp)
        pareto_overall_acc_eod = self.pareto_analyzer.identify_pareto_optimal_by_model_type(points_overall_acc_eod)
        pareto_overall_acc_eop = self.pareto_analyzer.identify_pareto_optimal_by_model_type(points_overall_acc_eop)
        
        Visualizer.plot_pareto_front_3subplots(
            points_overall_acc_dp, points_overall_acc_eod, points_overall_acc_eop,
            "Overall Pareto Front (Aggregated Accuracy)",
            r"Accuracy $\uparrow$",
            str(self.output_dir / "overall_accuracy"),
            method_colors=method_colors,
            pareto_dp=pareto_overall_acc_dp,
            pareto_eod=pareto_overall_acc_eod,
            pareto_eop=pareto_overall_acc_eop
        )
        
        # Overall AUC plots
        points_overall_auc_dp = [ParetoPoint(r.checkpoint_name, r.auc_mean, r.dp_diff_mean, "Overall",
                                             base_model_name=r.base_model_name, model_type=r.model_type)
                                for r in self.aggregated_results]
        points_overall_auc_eod = [ParetoPoint(r.checkpoint_name, r.auc_mean, r.eod_diff_mean, "Overall",
                                              base_model_name=r.base_model_name, model_type=r.model_type)
                                 for r in self.aggregated_results]
        points_overall_auc_eop = [ParetoPoint(r.checkpoint_name, r.auc_mean, r.eop_diff_mean, "Overall",
                                              base_model_name=r.base_model_name, model_type=r.model_type)
                                 for r in self.aggregated_results]
        
        pareto_overall_auc_dp = self.pareto_analyzer.identify_pareto_optimal_by_model_type(points_overall_auc_dp)
        pareto_overall_auc_eod = self.pareto_analyzer.identify_pareto_optimal_by_model_type(points_overall_auc_eod)
        pareto_overall_auc_eop = self.pareto_analyzer.identify_pareto_optimal_by_model_type(points_overall_auc_eop)
        
        Visualizer.plot_pareto_front_3subplots(
            points_overall_auc_dp, points_overall_auc_eod, points_overall_auc_eop,
            "Overall Pareto Front (Aggregated ROCAUC)",
            r"ROCAUC $\uparrow$",
            str(self.output_dir / "overall_auc"),
            method_colors=method_colors,
            pareto_dp=pareto_overall_auc_dp,
            pareto_eod=pareto_overall_auc_eod,
            pareto_eop=pareto_overall_auc_eop
        )
    
    def save_results_to_csv(self):
        """Save evaluation results and aggregated metrics to CSV."""
        print("\n" + "="*70)
        print("SAVING RESULTS TO CSV")
        print("="*70)
        
        # Detailed results
        detailed_df = pd.DataFrame([asdict(r) for r in self.evaluation_results])
        detailed_path = self.output_dir / "detailed_results.csv"
        detailed_df.to_csv(detailed_path, index=False)
        print(f"✓ Detailed results: {detailed_path}")
        
        # Aggregated results
        aggregated_df = pd.DataFrame([asdict(r) for r in self.aggregated_results])
        aggregated_path = self.output_dir / "aggregated_results.csv"
        aggregated_df.to_csv(aggregated_path, index=False)
        print(f"✓ Aggregated results: {aggregated_path}")
    
    def save_evaluation_results(self):
        """Save evaluation results to pickle for fast reloading."""
        results_file = self.output_dir / "evaluation_results.pkl"
        data = {
            "evaluation_results": self.evaluation_results,
            "aggregated_results": self.aggregated_results,
            "model_types": self.model_types
        }
        with open(results_file, "wb") as f:
            pickle.dump(data, f)
        print(f"✓ Saved evaluation results: {results_file}")
    
    def load_evaluation_results(self) -> bool:
        """Load evaluation results from pickle. Returns True if successful, False if file not found."""
        results_file = self.output_dir / "evaluation_results.pkl"
        if not results_file.exists():
            print(f"⚠ Results file not found: {results_file}")
            return False
        
        try:
            with open(results_file, "rb") as f:
                data = pickle.load(f)
            self.evaluation_results = data["evaluation_results"]
            self.aggregated_results = data["aggregated_results"]
            self.model_types = data["model_types"]
            print(f"✓ Loaded {len(self.evaluation_results)} evaluation results from {results_file}")
            return True
        except Exception as e:
            print(f"✗ Error loading results: {e}")
            return False
    
    def regenerate_plots_only(self):
        """Regenerate plots from previously saved evaluation results."""
        print("\n" + "="*70)
        print("REGENERATING PLOTS FROM SAVED RESULTS")
        print("="*70)
        
        if not self.load_evaluation_results():
            print("Cannot regenerate plots without evaluation results.")
            return
        
        # Generate Pareto fronts
        self.generate_pareto_fronts()
        
        # Generate visualizations
        self.visualize_pareto_fronts()
        
        print("\n" + "="*70)
        print("PLOT REGENERATION COMPLETE")
        print(f"Plots saved to: {self.output_dir}")
        print("="*70)
    
    def run_full_pipeline(self):
        """Execute the complete analysis pipeline."""
        print("\n" + "="*70)
        print("STARTING CHECKPOINT AGGREGATION & PARETO ANALYSIS PIPELINE")
        print("="*70)
        
        # Step 1: Evaluate all checkpoints
        self.evaluate_all_checkpoints()
        
        # Step 2: Aggregate results
        self.aggregate_results()
        
        # Step 2b: Save results for fast re-plotting
        self.save_evaluation_results()
        
        # Step 3: Build Pareto fronts
        self.generate_pareto_fronts()
        
        # Step 4: Generate visualizations
        self.visualize_pareto_fronts()
        
        # Step 5: Save results to CSV
        self.save_results_to_csv()
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print(f"Results saved to: {self.output_dir}")
        print("="*70)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Model Evaluation and Pareto Front Analysis for Fair ML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Regenerate plots from saved results (default, fast)
  python pareto_checkpoint_analysis.py
  
  # Include per-dataset plots in regeneration
  python pareto_checkpoint_analysis.py --per-dataset-plots
  
  # Run full evaluation pipeline from scratch
  python pareto_checkpoint_analysis.py --full-eval
  
  # Run full evaluation and generate per-dataset plots
  python pareto_checkpoint_analysis.py --full-eval --per-dataset-plots
        """
    )
    
    parser.add_argument(
        "--per-dataset-plots",
        action="store_true",
        default=False,
        help="Generate per-dataset Pareto plots in addition to overall plots (default: False)"
    )
    
    parser.add_argument(
        "--full-eval",
        action="store_true",
        default=False,
        help="Run full evaluation pipeline from scratch instead of regenerating plots from saved results (default: False, i.e., regen plots only)"
    )
    
    parser.add_argument(
        "--csv-path",
        type=str,
        default="eval_config/eval_models.csv",
        help="Path to eval_models.csv"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./eval_results/pareto_analysis",
        help="Output directory for results and plots"
    )
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize the pipeline
    pipeline = CheckpointAnalysisPipeline(
        eval_models_csv=args.csv_path,
        output_dir=args.output_dir,
        seed=42,
        generate_per_dataset_plots=args.per_dataset_plots
    )
    
    if args.full_eval:
        # Run full pipeline (evaluate all checkpoints + generate plots)
        pipeline.run_full_pipeline()
    else:
        # Regenerate plots from previously saved results (default, much faster)
        pipeline.regenerate_plots_only()
