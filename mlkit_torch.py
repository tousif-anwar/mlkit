"""
MLKit PyTorch: One-Liner ML/DL/CV Library for PyTorch
Like lenskit but for PyTorch-based Machine Learning & Deep Learning

Usage:
    import mlkit_torch as mlkit
    mlkit.eval.evaluate(y_true, y_pred)
    mlkit.viz.confusion(y_true, y_pred)
    mlkit.torch.train_step(model, x, y, optimizer, loss_fn)
    mlkit.torch.validate(model, val_loader)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
import pickle
from abc import ABC, abstractmethod
import warnings
from collections import defaultdict
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from scipy import stats
from scipy.ndimage import gaussian_filter

warnings.filterwarnings('ignore')

# ============================================================================
# DEVICE UTILITIES
# ============================================================================

class device:
    """One-liner device management"""
    
    @staticmethod
    def get() -> torch.device:
        """One-liner: mlkit.device.get()"""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @staticmethod
    def info() -> Dict:
        """One-liner: mlkit.device.info()"""
        dev = device.get()
        info = {'device': str(dev), 'cuda_available': torch.cuda.is_available()}
        
        if torch.cuda.is_available():
            info['cuda_version'] = torch.version.cuda
            info['device_count'] = torch.cuda.device_count()
            info['device_name'] = torch.cuda.get_device_name(0)
            info['device_memory'] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB"
        
        return info
    
    @staticmethod
    def print_info() -> None:
        """One-liner: mlkit.device.print_info()"""
        print("\n" + "="*50)
        print("DEVICE INFORMATION")
        print("="*50)
        for key, value in device.info().items():
            print(f"{key:20}: {value}")
        print("="*50 + "\n")


# ============================================================================
# EVALUATION MODULE - One-liner evaluation
# ============================================================================

class eval:
    """One-liner evaluation functions"""
    
    @staticmethod
    def evaluate(y_true: Union[np.ndarray, torch.Tensor], 
                y_pred: Union[np.ndarray, torch.Tensor], 
                task: str = 'auto') -> Dict:
        """
        One-liner: mlkit.eval.evaluate(y_true, y_pred)
        
        Comprehensive evaluation with auto-detection of task type.
        """
        y_true = eval._to_numpy(y_true)
        y_pred = eval._to_numpy(y_pred)
        
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
            if y_pred.ndim > 1:
                y_pred_class = np.argmax(y_pred, axis=1)
            else:
                y_pred_class = (y_pred > 0.5).astype(int)
            results.update(eval._classification_metrics(y_true, y_pred_class))
        
        return results
    
    @staticmethod
    def _to_numpy(x):
        """Convert tensor to numpy"""
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)
    
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
            roc_auc_score
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
        """One-liner: mlkit.eval.compare({...})"""
        return pd.DataFrame(results).T.sort_values('accuracy', ascending=False)
    
    @staticmethod
    def report(y_true: Union[np.ndarray, torch.Tensor], 
              y_pred: Union[np.ndarray, torch.Tensor]) -> None:
        """One-liner: mlkit.eval.report(y_true, y_pred)"""
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
# VISUALIZATION MODULE
# ============================================================================

class viz:
    """One-liner visualization functions"""
    
    _style_applied = False
    
    @staticmethod
    def _apply_style():
        """Apply consistent styling"""
        if not viz._style_applied:
            sns.set_style("whitegrid")
            sns.set_palette("husl")
            plt.rcParams['figure.figsize'] = (12, 6)
            plt.rcParams['font.size'] = 10
            viz._style_applied = True
    
    @staticmethod
    def compare_models(results: Dict[str, Dict], metric: str = 'accuracy', 
                      save: str = None, show: bool = True) -> None:
        """One-liner: mlkit.viz.compare_models(results)"""
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
    def confusion(y_true: Union[np.ndarray, torch.Tensor], 
                 y_pred: Union[np.ndarray, torch.Tensor], 
                 labels: List = None, save: str = None, show: bool = True) -> None:
        """One-liner: mlkit.viz.confusion(y_true, y_pred)"""
        viz._apply_style()
        from sklearn.metrics import confusion_matrix
        
        y_true = eval._to_numpy(y_true)
        y_pred = eval._to_numpy(y_pred)
        
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                   xticklabels=labels, yticklabels=labels)
        axes[0].set_title('Confusion Matrix (Counts)')
        axes[0].set_ylabel('True')
        axes[0].set_xlabel('Predicted')
        
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
        """One-liner: mlkit.viz.history(history)"""
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
    def roc(y_true: Union[np.ndarray, torch.Tensor], 
           y_pred: Union[np.ndarray, torch.Tensor],
           save: str = None, show: bool = True) -> None:
        """One-liner: mlkit.viz.roc(y_true, y_pred)"""
        viz._apply_style()
        from sklearn.metrics import roc_curve, auc
        
        y_true = eval._to_numpy(y_true)
        y_pred = eval._to_numpy(y_pred)
        
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
        """One-liner: mlkit.viz.learning_curve(train_scores, val_scores)"""
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
    def distributions(data: Union[np.ndarray, torch.Tensor, pd.DataFrame], 
                     labels: str = None, save: str = None, show: bool = True) -> None:
        """One-liner: mlkit.viz.distributions(data)"""
        viz._apply_style()
        
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        
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
    def predictions(y_true: Union[np.ndarray, torch.Tensor], 
                   y_pred: Union[np.ndarray, torch.Tensor],
                   save: str = None, show: bool = True) -> None:
        """One-liner: mlkit.viz.predictions(y_true, y_pred)"""
        viz._apply_style()
        
        y_true = eval._to_numpy(y_true)
        y_pred = eval._to_numpy(y_pred)
        
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
# DATA MODULE
# ============================================================================

class data:
    """One-liner data utilities"""
    
    @staticmethod
    def analyze(dataset: Union[np.ndarray, torch.Tensor, pd.DataFrame]) -> Dict:
        """One-liner: mlkit.data.analyze(X)"""
        if isinstance(dataset, torch.Tensor):
            dataset = dataset.detach().cpu().numpy()
        
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
            }
    
    @staticmethod
    def summary(dataset: Union[np.ndarray, torch.Tensor, pd.DataFrame]) -> None:
        """One-liner: mlkit.data.summary(X)"""
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
    def split(X: Union[np.ndarray, torch.Tensor], 
             y: Union[np.ndarray, torch.Tensor],
             test_size: float = 0.2, val_size: float = 0.1,
             random_state: int = 42) -> Dict:
        """One-liner: mlkit.data.split(X, y)"""
        from sklearn.model_selection import train_test_split
        
        X = eval._to_numpy(X)
        y = eval._to_numpy(y)
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
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
    def normalize(X: Union[np.ndarray, torch.Tensor], 
                 method: str = 'zscore') -> Tuple[Union[np.ndarray, torch.Tensor], Dict]:
        """One-liner: mlkit.data.normalize(X)"""
        is_tensor = isinstance(X, torch.Tensor)
        X_np = eval._to_numpy(X)
        
        if method == 'minmax':
            min_vals = X_np.min(axis=0)
            max_vals = X_np.max(axis=0)
            X_norm = (X_np - min_vals) / (max_vals - min_vals + 1e-8)
            params = {'method': 'minmax', 'min': min_vals, 'max': max_vals}
        elif method == 'zscore':
            mean = X_np.mean(axis=0)
            std = X_np.std(axis=0)
            X_norm = (X_np - mean) / (std + 1e-8)
            params = {'method': 'zscore', 'mean': mean, 'std': std}
        else:
            raise ValueError("method must be 'minmax' or 'zscore'")
        
        if is_tensor:
            X_norm = torch.from_numpy(X_norm).float()
        
        return X_norm, params
    
    @staticmethod
    def to_tensor(X: np.ndarray, y: np.ndarray = None, 
                 dtype: torch.dtype = torch.float32) -> Union[torch.Tensor, Tuple]:
        """One-liner: mlkit.data.to_tensor(X, y)"""
        X_tensor = torch.from_numpy(X).to(dtype)
        
        if y is not None:
            y_np = eval._to_numpy(y)
            y_tensor = torch.from_numpy(y_np).long() if y_np.dtype == np.int64 else torch.from_numpy(y_np).float()
            return X_tensor, y_tensor
        
        return X_tensor
    
    @staticmethod
    def loader(X: Union[np.ndarray, torch.Tensor], 
              y: Union[np.ndarray, torch.Tensor] = None,
              batch_size: int = 32, shuffle: bool = True,
              num_workers: int = 0) -> DataLoader:
        """One-liner: mlkit.data.loader(X, y, batch_size=32)"""
        X_tensor = torch.from_numpy(eval._to_numpy(X)).float() if not isinstance(X, torch.Tensor) else X
        
        if y is not None:
            y_tensor = torch.from_numpy(eval._to_numpy(y)).long() if not isinstance(y, torch.Tensor) else y
            dataset = TensorDataset(X_tensor, y_tensor)
        else:
            dataset = TensorDataset(X_tensor)
        
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


# ============================================================================
# PYTORCH TRAINING MODULE
# ============================================================================

class torch_train:
    """One-liner PyTorch training utilities"""
    
    @staticmethod
    def step(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
            optimizer: optim.Optimizer, loss_fn: Callable,
            device: torch.device = None) -> float:
        """
        One-liner: loss = mlkit.torch_train.step(model, x, y, optimizer, loss_fn)
        
        Single training step
        """
        if device is None:
            device = next(model.parameters()).device
        
        model.train()
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        
        return float(loss.item())
    
    @staticmethod
    def batch(model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer,
             loss_fn: Callable, device: torch.device = None) -> float:
        """
        One-liner: avg_loss = mlkit.torch_train.batch(model, loader, optimizer, loss_fn)
        
        Train for one epoch
        """
        if device is None:
            device = next(model.parameters()).device
        
        model.train()
        total_loss = 0.0
        n_batches = 0
        
        for batch in loader:
            if len(batch) == 2:
                x, y = batch
            else:
                x = batch[0]
                y = None
            
            x = x.to(device)
            if y is not None:
                y = y.to(device)
            
            optimizer.zero_grad()
            y_pred = model(x)
            
            if y is not None:
                loss = loss_fn(y_pred, y)
            else:
                loss = loss_fn(y_pred)
            
            loss.backward()
            optimizer.step()
            
            total_loss += float(loss.item())
            n_batches += 1
        
        return total_loss / max(n_batches, 1)
    
    @staticmethod
    def validate(model: nn.Module, loader: DataLoader, loss_fn: Callable,
                device: torch.device = None) -> Dict[str, float]:
        """
        One-liner: metrics = mlkit.torch_train.validate(model, val_loader, loss_fn)
        
        Validate model
        """
        if device is None:
            device = next(model.parameters()).device
        
        model.eval()
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for batch in loader:
                if len(batch) == 2:
                    x, y = batch
                else:
                    x = batch[0]
                    y = None
                
                x = x.to(device)
                if y is not None:
                    y = y.to(device)
                
                y_pred = model(x)
                
                if y is not None:
                    loss = loss_fn(y_pred, y)
                else:
                    loss = loss_fn(y_pred)
                
                total_loss += float(loss.item())
                n_batches += 1
        
        return {'val_loss': total_loss / max(n_batches, 1)}
    
    @staticmethod
    def predict(model: nn.Module, loader: DataLoader,
               device: torch.device = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        One-liner: y_pred, y_true = mlkit.torch_train.predict(model, test_loader)
        
        Get predictions on dataset
        """
        if device is None:
            device = next(model.parameters()).device
        
        model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in loader:
                if len(batch) == 2:
                    x, y = batch
                    targets.append(y.numpy())
                else:
                    x = batch[0]
                
                x = x.to(device)
                y_pred = model(x)
                predictions.append(y_pred.detach().cpu().numpy())
        
        y_pred = np.vstack(predictions)
        y_true = np.hstack(targets) if targets else None
        
        return y_pred, y_true
    
    @staticmethod
    def fit(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
           optimizer: optim.Optimizer, loss_fn: Callable, epochs: int = 10,
           device: torch.device = None, patience: int = None) -> Dict:
        """
        One-liner: history = mlkit.torch_train.fit(model, train_loader, val_loader, optimizer, loss_fn)
        
        Complete training loop
        """
        if device is None:
            device = next(model.parameters()).device
        
        history = defaultdict(list)
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
            train_loss = torch_train.batch(model, train_loader, optimizer, loss_fn, device)
            history['train_loss'].append(train_loss)
            
            # Validate
            val_metrics = torch_train.validate(model, val_loader, loss_fn, device)
            val_loss = val_metrics['val_loss']
            history['val_loss'].append(val_loss)
            
            # Early stopping
            if patience is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch}")
                        break
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{epochs} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}")
        
        return dict(history)
    
    @staticmethod
    def lr_schedule(epoch: int, initial_lr: float, schedule_type: str = 'exponential',
                   decay_rate: float = 0.95) -> float:
        """One-liner: new_lr = mlkit.torch_train.lr_schedule(epoch, 0.001)"""
        if schedule_type == 'exponential':
            return initial_lr * (decay_rate ** (epoch // 10))
        elif schedule_type == 'linear':
            return initial_lr * (1 - epoch / 100)
        elif schedule_type == 'cosine':
            return initial_lr * (1 + np.cos(np.pi * epoch / 100)) / 2
        else:
            return initial_lr
    
    @staticmethod
    def update_lr(optimizer: optim.Optimizer, new_lr: float) -> None:
        """One-liner: mlkit.torch_train.update_lr(optimizer, new_lr)"""
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


# ============================================================================
# COMPUTER VISION MODULE
# ============================================================================

class vision:
    """One-liner computer vision utilities"""
    
    @staticmethod
    def view_batch(images: Union[np.ndarray, torch.Tensor], labels: List = None,
                  grid_size: Tuple = (4, 4), save: str = None, show: bool = True) -> None:
        """One-liner: mlkit.vision.view_batch(images, labels=[...])"""
        viz._apply_style()
        
        images = eval._to_numpy(images)
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
                # Normalize if needed
                if img.max() <= 1:
                    ax.imshow(img)
                else:
                    ax.imshow(img.astype(np.uint8))
            else:
                ax.imshow(img, cmap='gray')
            
            if labels is not None:
                ax.set_title(labels[idx])
            ax.axis('off')
        
        plt.tight_layout()
        if save:
            plt.savefig(save, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
    
    @staticmethod
    def saliency_map(image: Union[np.ndarray, torch.Tensor], save: str = None,
                    show: bool = True) -> None:
        """One-liner: mlkit.vision.saliency_map(image)"""
        viz._apply_style()
        
        image = eval._to_numpy(image)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Simple saliency: edge detection
        if image.ndim == 3 and image.shape[2] == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image.squeeze()
        
        saliency = gaussian_filter(np.abs(np.gradient(gray)[0]) + 
                                  np.abs(np.gradient(gray)[1]), sigma=2)
        
        axes[1].imshow(saliency, cmap='hot')
        axes[1].set_title('Saliency Map')
        axes[1].axis('off')
        
        plt.tight_layout()
        if save:
            plt.savefig(save, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
    
    @staticmethod
    def feature_maps(features: Union[np.ndarray, torch.Tensor], 
                    n_maps: int = 16, save: str = None, show: bool = True) -> None:
        """One-liner: mlkit.vision.feature_maps(features)"""
        viz._apply_style()
        
        features = eval._to_numpy(features)
        
        if features.ndim == 4:  # [batch, channels, height, width]
            features = features[0]  # First sample
        
        n_maps = min(n_maps, features.shape[0])
        n_cols = 4
        n_rows = (n_maps + 3) // 4
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3*n_rows))
        axes = axes.flatten()
        
        for idx in range(n_maps):
            ax = axes[idx]
            feature_map = features[idx]
            ax.imshow(feature_map, cmap='viridis')
            ax.set_title(f'Feature {idx}')
            ax.axis('off')
        
        for idx in range(n_maps, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        if save:
            plt.savefig(save, dpi=300, bbox_inches='tight')
        if show:
            plt.show()


# ============================================================================
# RESULTS MODULE
# ============================================================================

class results:
    """One-liner results management"""
    
    @staticmethod
    def save(data: Dict, filepath: str, format: str = 'json') -> None:
        """One-liner: mlkit.results.save(data, 'results.json')"""
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
        
        print(f"Results saved to {filepath}")
    
    @staticmethod
    def load(filepath: str, format: str = None) -> Dict:
        """One-liner: mlkit.results.load('results.json')"""
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
        """One-liner: mlkit.results.summary(data)"""
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
# COMPARISON MODULE
# ============================================================================

class compare:
    """One-liner comparison functions"""
    
    @staticmethod
    def models(results_dict: Dict[str, Dict], metrics: List = None) -> pd.DataFrame:
        """One-liner: mlkit.compare.models(results_dict)"""
        df = pd.DataFrame(results_dict).T
        
        if metrics:
            df = df[metrics]
        
        return df.sort_values(df.columns[0], ascending=False)
    
    @staticmethod
    def datasets(dataset_results: Dict[str, Dict]) -> pd.DataFrame:
        """One-liner: mlkit.compare.datasets(dataset_results)"""
        return pd.DataFrame(dataset_results).T
    
    @staticmethod
    def hyperparams(hp_results: Dict[str, Dict], metric: str = 'accuracy',
                   save: str = None, show: bool = True) -> None:
        """One-liner: mlkit.compare.hyperparams(hp_results)"""
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
        if show:
            plt.show()


# ============================================================================
# MODEL UTILITIES
# ============================================================================

class models:
    """One-liner model utilities"""
    
    @staticmethod
    def count_params(model: nn.Module) -> Dict[str, int]:
        """One-liner: mlkit.models.count_params(model)"""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {'total': total, 'trainable': trainable}
    
    @staticmethod
    def summary(model: nn.Module, input_size: Tuple) -> None:
        """One-liner: mlkit.models.summary(model, (3, 224, 224))"""
        from torchsummary import summary as torch_summary
        try:
            torch_summary(model, input_size)
        except:
            print("Install torchsummary: pip install torchsummary")
    
    @staticmethod
    def freeze(model: nn.Module, freeze: bool = True) -> None:
        """One-liner: mlkit.models.freeze(model)"""
        for param in model.parameters():
            param.requires_grad = not freeze
    
    @staticmethod
    def save(model: nn.Module, filepath: str) -> None:
        """One-liner: mlkit.models.save(model, 'model.pt')"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), filepath)
        print(f"Model saved to {filepath}")
    
    @staticmethod
    def load(model: nn.Module, filepath: str, device: torch.device = None) -> nn.Module:
        """One-liner: mlkit.models.load(model, 'model.pt')"""
        if device is None:
            device = next(model.parameters()).device
        
        model.load_state_dict(torch.load(filepath, map_location=device))
        return model


# ============================================================================
# UTILITIES
# ============================================================================

def seed(value: int = 42) -> None:
    """One-liner: mlkit.seed(42)"""
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)
    try:
        import tensorflow as tf
        tf.random.set_seed(value)
    except:
        pass


def info() -> Dict:
    """One-liner: mlkit.info()"""
    info_dict = {
        'numpy': np.__version__,
        'pandas': pd.__version__,
        'torch': torch.__version__,
        'cuda': torch.cuda.is_available()
    }
    
    try:
        from sklearn import __version__ as sklearn_version
        info_dict['sklearn'] = sklearn_version
    except:
        pass
    
    return info_dict


def print_info() -> None:
    """One-liner: mlkit.print_info()"""
    print("\n" + "="*50)
    print("MLKIT PYTORCH - ENVIRONMENT INFO")
    print("="*50)
    for key, value in info().items():
        print(f"{key:20}: {value}")
    device.print_info()


if __name__ == "__main__":
    print(__doc__)
    print_info()
