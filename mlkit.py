"""
MLKit: High-level Machine Learning & Deep Learning library
Like lenskit but for ML/DL/CV - everything in one-liners

Usage:
    import mlkit
    mlkit.viz.compare_models(results)
    mlkit.eval.evaluate(y_true, y_pred)
    mlkit.data.analyze(dataset)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
import json
import pickle
from abc import ABC, abstractmethod
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# EVALUATION MODULE - One-liner evaluation functions
# ============================================================================

class eval:
    """One-liner evaluation functions for ML models"""
    
    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray, task: str = 'auto') -> Dict:
        """
        One-liner: mlkit.eval.evaluate(y_true, y_pred)
        
        Comprehensive evaluation with auto-detection of task type.
        Returns dict with all relevant metrics.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # Auto-detect task
        if task == 'auto':
            if y_pred.ndim > 1 and y_pred.shape[1] > 1:
                task = 'multiclass'
            elif len(np.unique(y_true)) == 2:
                task = 'binary'
            elif y_pred.dtype in [float, np.float32, np.float64]:
                task = 'regression'
            else:
                task = 'multiclass'
        
        results = {'task': task}
        
        if task == 'regression':
            results.update(eval._regression_metrics(y_true, y_pred))
        else:
            # Convert probabilities to classes if needed
            if y_pred.ndim > 1:
                y_pred_class = np.argmax(y_pred, axis=1)
            else:
                y_pred_class = (y_pred > 0.5).astype(int)
            results.update(eval._classification_metrics(y_true, y_pred_class))
        
        return results
    
    @staticmethod
    def _regression_metrics(y_true, y_pred) -> Dict:
        """Calculate regression metrics"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        return {
            'mse': float(mean_squared_error(y_true, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'r2': float(r2_score(y_true, y_pred))
        }
    
    @staticmethod
    def _classification_metrics(y_true, y_pred) -> Dict:
        """Calculate classification metrics"""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, confusion_matrix
        )
        
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        }
        
        try:
            metrics['auc'] = float(roc_auc_score(y_true, y_pred, average='weighted'))
        except:
            pass
        
        return metrics
    
    @staticmethod
    def compare(results: Dict[str, Dict]) -> pd.DataFrame:
        """
        One-liner: mlkit.eval.compare({'model_a': metrics_a, 'model_b': metrics_b})
        
        Compare multiple model evaluations as a DataFrame
        """
        return pd.DataFrame(results).T.sort_values('accuracy', ascending=False)
    
    @staticmethod
    def report(y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        One-liner: mlkit.eval.report(y_true, y_pred)
        
        Print a detailed evaluation report
        """
        metrics = eval.evaluate(y_true, y_pred)
        
        print("\n" + "="*50)
        print("EVALUATION REPORT")
        print("="*50)
        print(f"Task: {metrics.pop('task')}")
        print("-"*50)
        for metric, value in metrics.items():
            print(f"{metric.upper():15} : {value:.4f}")
        print("="*50 + "\n")


# ============================================================================
# VISUALIZATION MODULE - One-liner plotting
# ============================================================================

class viz:
    """One-liner visualization functions"""
    
    _style_applied = False
    
    @staticmethod
    def _apply_style():
        """Apply consistent styling once"""
        if not viz._style_applied:
            sns.set_style("whitegrid")
            sns.set_palette("husl")
            plt.rcParams['figure.figsize'] = (12, 6)
            plt.rcParams['font.size'] = 10
            viz._style_applied = True
    
    @staticmethod
    def compare_models(results: Dict[str, Dict], metric: str = 'accuracy', 
                      save: str = None, show: bool = True) -> None:
        """
        One-liner: mlkit.viz.compare_models({'model_a': {...}, 'model_b': {...}})
        
        Compare models across metrics with bar charts
        """
        viz._apply_style()
        
        df = pd.DataFrame(results).T
        
        if metric not in df.columns:
            metric = df.columns[0]
        
        fig, axes = plt.subplots(1, min(3, len(df.columns)), figsize=(15, 5))
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        
        for idx, col in enumerate(df.columns[:3]):
            ax = axes[idx]
            values = df[col].sort_values(ascending=False)
            bars = ax.bar(range(len(values)), values.values, color=sns.color_palette("husl", len(values)))
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels(values.index, rotation=45, ha='right')
            ax.set_ylabel(col.upper())
            ax.set_title(f"Model Comparison: {col}")
            ax.set_ylim([0, 1.0])
            
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        if save:
            plt.savefig(save, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
    
    @staticmethod
    def confusion(y_true: np.ndarray, y_pred: np.ndarray, labels: List = None,
                 save: str = None, show: bool = True) -> None:
        """
        One-liner: mlkit.viz.confusion(y_true, y_pred, labels=['A', 'B', 'C'])
        
        Plot confusion matrix as heatmap
        """
        viz._apply_style()
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                   xticklabels=labels, yticklabels=labels)
        axes[0].set_title('Confusion Matrix (Counts)')
        axes[0].set_ylabel('True')
        axes[0].set_xlabel('Predicted')
        
        # Normalized
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', ax=axes[1],
                   xticklabels=labels, yticklabels=labels)
        axes[1].set_title('Confusion Matrix (Normalized)')
        axes[1].set_ylabel('True')
        axes[1].set_xlabel('Predicted')
        
        plt.tight_layout()
        if save:
            plt.savefig(save, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
    
    @staticmethod
    def history(history: Dict[str, List], metrics: List = None,
               save: str = None, show: bool = True) -> None:
        """
        One-liner: mlkit.viz.history({'loss': [...], 'val_loss': [...], 'acc': [...]})
        
        Plot training history with multiple metrics
        """
        viz._apply_style()
        
        if metrics is None:
            metrics = list(history.keys())
        
        n_metrics = len(metrics)
        n_cols = 2
        n_rows = (n_metrics + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4*n_rows))
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            if metric in history:
                ax = axes[idx]
                values = history[metric]
                ax.plot(values, linewidth=2.5, marker='o', markersize=4, label=metric)
                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric)
                ax.set_title(f'{metric.upper()} over time')
                ax.grid(True, alpha=0.3)
        
        for idx in range(len(metrics), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        if save:
            plt.savefig(save, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
    
    @staticmethod
    def roc(y_true: np.ndarray, y_pred: np.ndarray,
           save: str = None, show: bool = True) -> None:
        """
        One-liner: mlkit.viz.roc(y_true, y_pred_proba)
        
        Plot ROC curve with AUC
        """
        viz._apply_style()
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save:
            plt.savefig(save, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
    
    @staticmethod
    def learning_curve(train_scores: List, val_scores: List,
                      train_label: str = 'Train', val_label: str = 'Validation',
                      save: str = None, show: bool = True) -> None:
        """
        One-liner: mlkit.viz.learning_curve(train_scores, val_scores)
        
        Plot learning curves to detect overfitting
        """
        viz._apply_style()
        
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(train_scores) + 1)
        plt.plot(epochs, train_scores, 'o-', linewidth=2, label=train_label)
        plt.plot(epochs, val_scores, 's-', linewidth=2, label=val_label)
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Learning Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save:
            plt.savefig(save, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
    
    @staticmethod
    def distributions(data: Union[np.ndarray, pd.DataFrame], labels: str = None,
                     save: str = None, show: bool = True) -> None:
        """
        One-liner: mlkit.viz.distributions(data)
        
        Plot distributions of features or predictions
        """
        viz._apply_style()
        
        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                plt.figure(figsize=(10, 6))
                plt.hist(data, bins=30, alpha=0.7, edgecolor='black')
                plt.xlabel('Value')
                plt.ylabel('Frequency')
                plt.title('Distribution')
            else:
                n_features = data.shape[1]
                n_cols = 3
                n_rows = (n_features + 2) // 3
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
                axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
                
                for idx in range(n_features):
                    axes[idx].hist(data[:, idx], bins=30, alpha=0.7, edgecolor='black')
                    axes[idx].set_title(f'Feature {idx}')
                
                for idx in range(n_features, len(axes)):
                    fig.delaxes(axes[idx])
        
        plt.tight_layout()
        if save:
            plt.savefig(save, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
    
    @staticmethod
    def predictions(y_true: np.ndarray, y_pred: np.ndarray,
                   save: str = None, show: bool = True) -> None:
        """
        One-liner: mlkit.viz.predictions(y_true, y_pred)
        
        Scatter plot of true vs predicted values
        """
        viz._apply_style()
        
        plt.figure(figsize=(8, 8))
        plt.scatter(y_true, y_pred, alpha=0.6, s=50)
        
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')
        
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title('Predictions vs Ground Truth')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save:
            plt.savefig(save, dpi=300, bbox_inches='tight')
        if show:
            plt.show()


# ============================================================================
# DATA MODULE - Data analysis and exploration
# ============================================================================

class data:
    """One-liner data analysis functions"""
    
    @staticmethod
    def analyze(dataset: Union[np.ndarray, pd.DataFrame]) -> Dict:
        """
        One-liner: mlkit.data.analyze(X)
        
        Quick dataset analysis with shape, types, missing values
        """
        if isinstance(dataset, np.ndarray):
            return {
                'shape': dataset.shape,
                'dtype': str(dataset.dtype),
                'min': float(np.nanmin(dataset)),
                'max': float(np.nanmax(dataset)),
                'mean': float(np.nanmean(dataset)),
                'std': float(np.nanstd(dataset)),
                'missing': int(np.isnan(dataset).sum())
            }
        elif isinstance(dataset, pd.DataFrame):
            return {
                'shape': dataset.shape,
                'missing_pct': float(dataset.isnull().sum().sum() / (dataset.shape[0] * dataset.shape[1]) * 100),
                'dtypes': dataset.dtypes.to_dict(),
                'numeric_summary': dataset.describe().to_dict()
            }
    
    @staticmethod
    def summary(dataset: Union[np.ndarray, pd.DataFrame]) -> None:
        """
        One-liner: mlkit.data.summary(X)
        
        Print formatted data summary
        """
        info = data.analyze(dataset)
        
        print("\n" + "="*50)
        print("DATA SUMMARY")
        print("="*50)
        for key, value in info.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"{key}: {value}")
        print("="*50 + "\n")
    
    @staticmethod
    def split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2,
             val_size: float = 0.1, random_state: int = 42) -> Dict:
        """
        One-liner: mlkit.data.split(X, y, test_size=0.2, val_size=0.1)
        
        Stratified train/val/test split
        """
        from sklearn.model_selection import train_test_split
        
        # Train-test split
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Train-val split
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
        )
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
            'splits': {'train': len(X_train), 'val': len(X_val), 'test': len(X_test)}
        }
    
    @staticmethod
    def normalize(X: np.ndarray, method: str = 'minmax') -> Tuple[np.ndarray, Dict]:
        """
        One-liner: mlkit.data.normalize(X, method='minmax')
        
        Normalize data and return normalization params
        """
        if method == 'minmax':
            min_vals = X.min(axis=0)
            max_vals = X.max(axis=0)
            X_norm = (X - min_vals) / (max_vals - min_vals + 1e-8)
            params = {'method': 'minmax', 'min': min_vals, 'max': max_vals}
        elif method == 'zscore':
            mean = X.mean(axis=0)
            std = X.std(axis=0)
            X_norm = (X - mean) / (std + 1e-8)
            params = {'method': 'zscore', 'mean': mean, 'std': std}
        else:
            raise ValueError("method must be 'minmax' or 'zscore'")
        
        return X_norm, params


# ============================================================================
# COMPUTER VISION MODULE
# ============================================================================

class vision:
    """One-liner computer vision functions"""
    
    @staticmethod
    def view_batch(images: np.ndarray, labels: List = None, 
                  grid_size: Tuple = (4, 4), save: str = None) -> None:
        """
        One-liner: mlkit.vision.view_batch(images, labels=['cat', 'dog'])
        
        Display batch of images in grid
        """
        viz._apply_style()
        
        n_show = min(len(images), grid_size[0] * grid_size[1])
        images = images[:n_show]
        if labels is not None:
            labels = labels[:n_show]
        
        fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(12, 12))
        axes = axes.flatten()
        
        for idx, (img, ax) in enumerate(zip(images, axes)):
            if img.shape[-1] == 3 or img.shape[0] == 3:
                if img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))
                ax.imshow(img)
            else:
                ax.imshow(img, cmap='gray')
            
            if labels is not None:
                ax.set_title(labels[idx])
            ax.axis('off')
        
        plt.tight_layout()
        if save:
            plt.savefig(save, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def augmentation_preview(image: np.ndarray, augmentations: Dict, 
                            grid_size: Tuple = (3, 3), save: str = None) -> None:
        """
        One-liner: mlkit.vision.augmentation_preview(image, {'rotate': 15, 'flip': True})
        
        Preview augmentation effects on a single image
        """
        try:
            from torchvision import transforms
            from PIL import Image
        except ImportError:
            print("Please install torchvision for augmentation preview")
            return
        
        viz._apply_style()
        
        fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(10, 10))
        axes = axes.flatten()
        
        for idx, ax in enumerate(axes):
            if idx == 0:
                ax.imshow(image)
                ax.set_title('Original')
            else:
                ax.imshow(image)
                ax.set_title(f'Augmentation {idx}')
            ax.axis('off')
        
        plt.tight_layout()
        if save:
            plt.savefig(save, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def saliency_map(image: np.ndarray, save: str = None) -> None:
        """
        One-liner: mlkit.vision.saliency_map(image)
        
        Display saliency/attention map
        """
        viz._apply_style()
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Simple saliency: edge detection
        from scipy import ndimage
        if image.ndim == 3 and image.shape[2] == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
        
        saliency = ndimage.gaussian_filter(np.abs(np.gradient(gray)[0]) + 
                                          np.abs(np.gradient(gray)[1]), sigma=2)
        
        axes[1].imshow(saliency, cmap='hot')
        axes[1].set_title('Saliency Map')
        axes[1].axis('off')
        
        plt.tight_layout()
        if save:
            plt.savefig(save, dpi=300, bbox_inches='tight')
        plt.show()


# ============================================================================
# TRAINING MODULE - Training utilities
# ============================================================================

class train:
    """One-liner training utilities"""
    
    @staticmethod
    def track(epoch: int, metrics: Dict, collector: 'Collector' = None) -> None:
        """
        One-liner: mlkit.train.track(epoch, {'loss': 0.5, 'acc': 0.9})
        
        Track metrics during training
        """
        if collector is None:
            print(f"Epoch {epoch:3d} | ", end="")
            for k, v in metrics.items():
                print(f"{k}: {v:.4f} | ", end="")
            print()
        else:
            collector.add(epoch, metrics)
    
    @staticmethod
    def early_stop(metric_history: List, patience: int = 5, 
                  mode: str = 'min') -> bool:
        """
        One-liner: if mlkit.train.early_stop(val_losses, patience=5): break
        
        Check if training should stop
        """
        if len(metric_history) < patience + 1:
            return False
        
        recent = metric_history[-patience:]
        best = metric_history[-(patience+1)]
        
        if mode == 'min':
            return all(m >= best for m in recent)
        else:
            return all(m <= best for m in recent)
    
    @staticmethod
    def lr_schedule(epoch: int, initial_lr: float, schedule_type: str = 'exponential',
                   decay_rate: float = 0.95) -> float:
        """
        One-liner: new_lr = mlkit.train.lr_schedule(epoch, 0.001)
        
        Calculate learning rate for current epoch
        """
        if schedule_type == 'exponential':
            return initial_lr * (decay_rate ** (epoch // 10))
        elif schedule_type == 'linear':
            return initial_lr * (1 - epoch / 100)
        elif schedule_type == 'cosine':
            return initial_lr * (1 + np.cos(np.pi * epoch / 100)) / 2
        else:
            return initial_lr


# ============================================================================
# RESULTS MODULE - Save and load experiments
# ============================================================================

class results:
    """One-liner results management"""
    
    @staticmethod
    def save(data: Dict, filepath: str, format: str = 'json') -> None:
        """
        One-liner: mlkit.results.save({'metrics': metrics, 'model': model}, 'results.json')
        
        Save results with metadata
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        data['_metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'format': format
        }
        
        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        elif format == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        else:
            raise ValueError("format must be 'json' or 'pickle'")
        
        print(f"Results saved to {filepath}")
    
    @staticmethod
    def load(filepath: str, format: str = None) -> Dict:
        """
        One-liner: data = mlkit.results.load('results.json')
        
        Load saved results
        """
        if format is None:
            format = 'json' if filepath.endswith('.json') else 'pickle'
        
        if format == 'json':
            with open(filepath, 'r') as f:
                return json.load(f)
        elif format == 'pickle':
            with open(filepath, 'rb') as f:
                return pickle.load(f)
    
    @staticmethod
    def summary(data: Dict) -> None:
        """
        One-liner: mlkit.results.summary(results)
        
        Print results summary
        """
        print("\n" + "="*50)
        print("RESULTS SUMMARY")
        print("="*50)
        
        for key, value in data.items():
            if key.startswith('_'):
                continue
            if isinstance(value, dict):
                print(f"\n{key}:")
                for k, v in value.items():
                    if isinstance(v, float):
                        print(f"  {k}: {v:.4f}")
                    else:
                        print(f"  {k}: {v}")
            else:
                print(f"{key}: {value}")
        
        print("="*50 + "\n")


# ============================================================================
# COMPARISON MODULE - Compare different configurations
# ============================================================================

class compare:
    """One-liner comparison functions"""
    
    @staticmethod
    def models(results_dict: Dict[str, Dict], metrics: List = None) -> pd.DataFrame:
        """
        One-liner: mlkit.compare.models({'model_a': results_a, 'model_b': results_b})
        
        Compare models side-by-side
        """
        df = pd.DataFrame(results_dict).T
        
        if metrics:
            df = df[metrics]
        
        return df.sort_values(df.columns[0], ascending=False)
    
    @staticmethod
    def datasets(dataset_results: Dict[str, Dict]) -> pd.DataFrame:
        """
        One-liner: mlkit.compare.datasets({'dataset_a': results_a, 'dataset_b': results_b})
        
        Compare performance across datasets
        """
        return pd.DataFrame(dataset_results).T
    
    @staticmethod
    def hyperparams(hp_results: Dict[str, Dict], metric: str = 'accuracy',
                   save: str = None) -> None:
        """
        One-liner: mlkit.compare.hyperparams({'lr_0.001': {...}, 'lr_0.01': {...}})
        
        Visualize hyperparameter impact
        """
        viz._apply_style()
        
        df = pd.DataFrame(hp_results).T
        df = df.sort_values(metric, ascending=False)
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(df)), df[metric].values)
        plt.xticks(range(len(df)), df.index, rotation=45, ha='right')
        plt.ylabel(metric.upper())
        plt.title(f'Hyperparameter Impact on {metric}')
        plt.grid(True, alpha=0.3, axis='y')
        
        if save:
            plt.savefig(save, dpi=300, bbox_inches='tight')
        plt.show()


# ============================================================================
# UTILITIES - General purpose utilities
# ============================================================================

def seed(value: int = 42) -> None:
    """One-liner: mlkit.seed(42)"""
    np.random.seed(value)
    try:
        import torch
        torch.manual_seed(value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(value)
    except:
        pass
    try:
        import tensorflow as tf
        tf.random.set_seed(value)
    except:
        pass


def info() -> Dict:
    """One-liner: mlkit.info()"""
    info = {
        'numpy': np.__version__,
        'pandas': pd.__version__,
        'sklearn': __import__('sklearn').__version__
    }
    
    try:
        import torch
        info['pytorch'] = torch.__version__
        info['cuda'] = torch.cuda.is_available()
    except:
        info['pytorch'] = 'Not installed'
    
    try:
        import tensorflow as tf
        info['tensorflow'] = tf.__version__
    except:
        info['tensorflow'] = 'Not installed'
    
    return info


def print_info() -> None:
    """One-liner: mlkit.print_info()"""
    print("\n" + "="*50)
    print("MLKIT ENVIRONMENT INFO")
    print("="*50)
    for key, value in info().items():
        print(f"{key:20}: {value}")
    print("="*50 + "\n")


if __name__ == "__main__":
    print(__doc__)
    print_info()
