"""
mlkit.py — Framework-Agnostic ML Toolkit

40+ one-liner utility functions for machine learning workflows.
Works with any framework: scikit-learn, XGBoost, PyTorch, TensorFlow, etc.

Modules:
    eval     — Evaluation metrics (classification & regression)
    viz      — Visualization helpers
    data     — Data preprocessing & transformation
    vision   — Image loading, resizing, augmentation
    train    — Model training, cross-validation, persistence
    results  — Results logging & summarisation
    compare  — Model comparison & feature ranking
"""

import csv
import json
import os
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report as _classification_report,
    confusion_matrix as _confusion_matrix,
    f1_score,
    log_loss as _log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve as _precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    cross_val_score,
    learning_curve as _learning_curve,
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_numpy(arr):
    """Convert framework-specific tensors/arrays to NumPy."""
    if isinstance(arr, np.ndarray):
        return arr
    if isinstance(arr, (pd.Series, pd.DataFrame)):
        return arr.to_numpy()
    # PyTorch
    if hasattr(arr, "detach"):
        return arr.detach().cpu().numpy()
    # TensorFlow / Keras
    if hasattr(arr, "numpy") and callable(arr.numpy):
        return arr.numpy()
    return np.asarray(arr)

# ============================================================================
# EVAL MODULE — Evaluation Metrics
# ============================================================================

def accuracy(y_true, y_pred):
    """Return accuracy score."""
    return accuracy_score(_to_numpy(y_true), _to_numpy(y_pred))


def precision(y_true, y_pred, **kwargs):
    """Return precision score (default: weighted average for multiclass)."""
    kwargs.setdefault("average", "weighted")
    kwargs.setdefault("zero_division", 0)
    return precision_score(_to_numpy(y_true), _to_numpy(y_pred), **kwargs)


def recall(y_true, y_pred, **kwargs):
    """Return recall score (default: weighted average for multiclass)."""
    kwargs.setdefault("average", "weighted")
    kwargs.setdefault("zero_division", 0)
    return recall_score(_to_numpy(y_true), _to_numpy(y_pred), **kwargs)


def f1(y_true, y_pred, **kwargs):
    """Return F1 score (default: weighted average for multiclass)."""
    kwargs.setdefault("average", "weighted")
    kwargs.setdefault("zero_division", 0)
    return f1_score(_to_numpy(y_true), _to_numpy(y_pred), **kwargs)


def roc_auc(y_true, y_proba, **kwargs):
    """Return ROC-AUC score."""
    return roc_auc_score(_to_numpy(y_true), _to_numpy(y_proba), **kwargs)


def confusion_mat(y_true, y_pred):
    """Return confusion matrix as a NumPy array."""
    return _confusion_matrix(_to_numpy(y_true), _to_numpy(y_pred))


def class_report(y_true, y_pred, **kwargs):
    """Return a classification report string."""
    kwargs.setdefault("zero_division", 0)
    return _classification_report(_to_numpy(y_true), _to_numpy(y_pred), **kwargs)


def mse(y_true, y_pred):
    """Return mean squared error."""
    return mean_squared_error(_to_numpy(y_true), _to_numpy(y_pred))


def rmse(y_true, y_pred):
    """Return root mean squared error."""
    return float(np.sqrt(mean_squared_error(_to_numpy(y_true), _to_numpy(y_pred))))


def mae(y_true, y_pred):
    """Return mean absolute error."""
    return mean_absolute_error(_to_numpy(y_true), _to_numpy(y_pred))


def r2(y_true, y_pred):
    """Return R² (coefficient of determination)."""
    return r2_score(_to_numpy(y_true), _to_numpy(y_pred))


def log_loss_score(y_true, y_proba, **kwargs):
    """Return log loss (cross-entropy loss)."""
    return _log_loss(_to_numpy(y_true), _to_numpy(y_proba), **kwargs)

# ============================================================================
# VIZ MODULE — Visualization
# ============================================================================

def plot_confusion_matrix(y_true, y_pred, labels=None, cmap="Blues",
                          title="Confusion Matrix", save_path=None):
    """Plot a confusion matrix heatmap and optionally save to *save_path*."""
    cm = _confusion_matrix(_to_numpy(y_true), _to_numpy(y_pred), labels=labels)
    fig, ax = plt.subplots()
    tick_labels = labels if labels is not None else True
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, xticklabels=tick_labels,
                yticklabels=tick_labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_roc_curve(y_true, y_proba, title="ROC Curve", save_path=None):
    """Plot a ROC curve and optionally save to *save_path*."""
    fpr, tpr, _ = roc_curve(_to_numpy(y_true), _to_numpy(y_proba))
    auc_val = roc_auc_score(_to_numpy(y_true), _to_numpy(y_proba))
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {auc_val:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_precision_recall(y_true, y_proba, title="Precision-Recall Curve",
                          save_path=None):
    """Plot a precision-recall curve and optionally save to *save_path*."""
    prec, rec, _ = _precision_recall_curve(_to_numpy(y_true), _to_numpy(y_proba))
    fig, ax = plt.subplots()
    ax.plot(rec, prec)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_feature_importance(importances, feature_names=None,
                            title="Feature Importance", top_n=None,
                            save_path=None):
    """Plot a horizontal bar chart of feature importances.

    *importances* can be a 1-D array or a fitted tree-based model that
    exposes ``feature_importances_``.
    """
    if hasattr(importances, "feature_importances_"):
        importances = importances.feature_importances_
    importances = _to_numpy(importances).ravel()
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(len(importances))]
    idx = np.argsort(importances)
    if top_n is not None:
        idx = idx[-top_n:]
    fig, ax = plt.subplots()
    ax.barh(np.array(feature_names)[idx], importances[idx])
    ax.set_title(title)
    ax.set_xlabel("Importance")
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_learning_curve(model, X, y, cv=5, scoring="accuracy",
                        title="Learning Curve", save_path=None):
    """Plot a learning curve for *model* and optionally save to *save_path*."""
    train_sizes, train_scores, val_scores = _learning_curve(
        model, X, y, cv=cv, scoring=scoring,
        train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1,
    )
    fig, ax = plt.subplots()
    ax.plot(train_sizes, train_scores.mean(axis=1), label="Train")
    ax.plot(train_sizes, val_scores.mean(axis=1), label="Validation")
    ax.set_xlabel("Training Size")
    ax.set_ylabel(scoring.capitalize())
    ax.set_title(title)
    ax.legend()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_correlation_matrix(df, method="pearson", cmap="coolwarm",
                            title="Correlation Matrix", save_path=None):
    """Plot a correlation heatmap for a DataFrame."""
    corr = df.select_dtypes(include=[np.number]).corr(method=method)
    fig, ax = plt.subplots(figsize=(max(6, len(corr.columns)), max(5, len(corr.columns) - 1)))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, ax=ax)
    ax.set_title(title)
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_distribution(data, column=None, bins=30,
                      title=None, save_path=None):
    """Plot a histogram with KDE for a column or array."""
    fig, ax = plt.subplots()
    values = data[column] if column is not None else _to_numpy(data)
    ax.hist(values, bins=bins, density=True, alpha=0.6, edgecolor="black")
    sns.kdeplot(values, ax=ax)
    ax.set_title(title or (f"Distribution of {column}" if column else "Distribution"))
    ax.set_xlabel(column if column else "Value")
    ax.set_ylabel("Density")
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return fig

# ============================================================================
# DATA MODULE — Data Preprocessing & Transformation
# ============================================================================

def split(X, y, test_size=0.2, random_state=42, **kwargs):
    """Split data into train/test sets. Returns (X_train, X_test, y_train, y_test)."""
    return train_test_split(X, y, test_size=test_size,
                            random_state=random_state, **kwargs)


def normalize(X):
    """Min-max normalize features to [0, 1]. Returns (transformed, scaler)."""
    scaler = MinMaxScaler()
    return scaler.fit_transform(_to_numpy(X).reshape(-1, 1) if _to_numpy(X).ndim == 1 else _to_numpy(X)), scaler


def standardize(X):
    """Z-score standardize features. Returns (transformed, scaler)."""
    scaler = StandardScaler()
    return scaler.fit_transform(_to_numpy(X).reshape(-1, 1) if _to_numpy(X).ndim == 1 else _to_numpy(X)), scaler


def fill_missing(df, strategy="mean"):
    """Fill missing values in a DataFrame.

    *strategy*: ``"mean"``, ``"median"``, ``"mode"``, or a scalar value.
    """
    df = df.copy()
    for col in df.columns:
        if df[col].isna().any():
            if strategy == "mean":
                df[col] = df[col].fillna(df[col].mean())
            elif strategy == "median":
                df[col] = df[col].fillna(df[col].median())
            elif strategy == "mode":
                df[col] = df[col].fillna(df[col].mode().iloc[0])
            else:
                df[col] = df[col].fillna(strategy)
    return df


def encode_labels(y):
    """Encode categorical labels to integers. Returns (encoded, encoder)."""
    le = LabelEncoder()
    return le.fit_transform(_to_numpy(y)), le


def one_hot_encode(df, columns=None):
    """One-hot encode specified columns (or all object columns)."""
    if columns is None:
        columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
    return pd.get_dummies(df, columns=columns)


def remove_outliers(df, column, method="iqr", threshold=1.5):
    """Remove outliers from *column* using IQR (default) or z-score method.

    *method*: ``"iqr"`` or ``"zscore"``
    """
    s = df[column]
    if method == "iqr":
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        mask = (s >= q1 - threshold * iqr) & (s <= q3 + threshold * iqr)
    elif method == "zscore":
        z = (s - s.mean()) / s.std()
        mask = z.abs() <= threshold
    else:
        raise ValueError(f"Unknown method: {method!r}")
    return df.loc[mask].reset_index(drop=True)


def balance_classes(X, y, random_state=42):
    """Balance classes by oversampling the minority class(es).

    Uses random oversampling (no external SMOTE dependency).
    Returns (X_balanced, y_balanced).
    """
    X = _to_numpy(X)
    y = _to_numpy(y)
    classes, counts = np.unique(y, return_counts=True)
    max_count = counts.max()
    rng = np.random.RandomState(random_state)
    X_parts, y_parts = [X], [y]
    for cls, cnt in zip(classes, counts):
        if cnt < max_count:
            idx = np.where(y == cls)[0]
            extra = rng.choice(idx, size=max_count - cnt, replace=True)
            X_parts.append(X[extra])
            y_parts.append(y[extra])
    return np.concatenate(X_parts), np.concatenate(y_parts)


def reduce_dims(X, n_components=2, random_state=42):
    """Reduce dimensionality via PCA. Returns (transformed, pca_model)."""
    pca = PCA(n_components=n_components, random_state=random_state)
    return pca.fit_transform(_to_numpy(X)), pca

# ============================================================================
# VISION MODULE — Image Helpers
# ============================================================================

def load_image(path):
    """Load an image from *path* as a NumPy array (RGB)."""
    from PIL import Image
    return np.array(Image.open(path).convert("RGB"))


def resize_image(img, size):
    """Resize *img* (NumPy array) to *(width, height)*.  Returns NumPy array."""
    from PIL import Image
    pil = Image.fromarray(np.uint8(img))
    return np.array(pil.resize(size))


def augment_image(img, flip_horizontal=True, flip_vertical=False,
                  brightness_delta=0.0):
    """Apply simple augmentations to a NumPy image array.

    Returns the augmented image as a NumPy array.
    """
    out = img.copy().astype(np.float32)
    if flip_horizontal:
        out = np.flip(out, axis=1)
    if flip_vertical:
        out = np.flip(out, axis=0)
    if brightness_delta != 0.0:
        out = np.clip(out + brightness_delta, 0, 255)
    return out.astype(np.uint8)


def to_tensor(img):
    """Convert a HWC uint8 image to a CHW float32 array normalised to [0, 1].

    This is framework-agnostic — returns a plain NumPy array that can be fed
    into PyTorch (``torch.from_numpy``) or TensorFlow.
    """
    arr = _to_numpy(img).astype(np.float32) / 255.0
    if arr.ndim == 3:
        arr = np.transpose(arr, (2, 0, 1))  # HWC → CHW
    return arr


def show_image(img, title=None, save_path=None):
    """Display an image using matplotlib and optionally save to *save_path*."""
    fig, ax = plt.subplots()
    arr = _to_numpy(img)
    # Handle CHW → HWC
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr.squeeze(axis=2)
    ax.imshow(arr.astype(np.uint8) if arr.max() > 1 else arr, cmap="gray" if arr.ndim == 2 else None)
    ax.axis("off")
    if title:
        ax.set_title(title)
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return fig

# ============================================================================
# TRAIN MODULE — Training Utilities
# ============================================================================

def cross_val(model, X, y, cv=5, scoring="accuracy"):
    """Return cross-validation scores as a NumPy array."""
    return cross_val_score(model, X, y, cv=cv, scoring=scoring)


def grid_search(model, param_grid, X, y, cv=5, scoring="accuracy",
                refit=True, **kwargs):
    """Run grid search and return the fitted ``GridSearchCV`` object."""
    gs = GridSearchCV(model, param_grid, cv=cv, scoring=scoring,
                      refit=refit, **kwargs)
    gs.fit(X, y)
    return gs


def train_model(model, X_train, y_train):
    """Fit *model* on training data and return it."""
    model.fit(X_train, y_train)
    return model


def predict(model, X):
    """Return predictions from *model*."""
    return model.predict(X)


def predict_proba(model, X):
    """Return probability predictions from *model*."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    raise AttributeError(f"{type(model).__name__} has no predict_proba method.")


def save_model(model, path):
    """Save *model* to disk using joblib."""
    joblib.dump(model, path)
    return path


def load_model(path):
    """Load a model from disk using joblib."""
    return joblib.load(path)

# ============================================================================
# RESULTS MODULE — Results Logging & Summarisation
# ============================================================================

def log_results(results, path, fmt="json"):
    """Persist *results* (dict) to a JSON or CSV file.

    *fmt*: ``"json"`` or ``"csv"``
    """
    if fmt == "json":
        mode = "a" if os.path.exists(path) else "w"
        with open(path, mode) as f:
            f.write(json.dumps(results) + "\n")
    elif fmt == "csv":
        file_exists = os.path.exists(path)
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(results)
    else:
        raise ValueError(f"Unknown format: {fmt!r}")
    return path


def summarize(y_true, y_pred, task="classification"):
    """Return a summary dict of common metrics.

    *task*: ``"classification"`` or ``"regression"``
    """
    y_true, y_pred = _to_numpy(y_true), _to_numpy(y_pred)
    if task == "classification":
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }
    if task == "regression":
        return {
            "mse": float(mean_squared_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
        }
    raise ValueError(f"Unknown task: {task!r}")


def to_dataframe(results):
    """Convert a dict or list of dicts into a pandas DataFrame."""
    if isinstance(results, dict):
        results = [results]
    return pd.DataFrame(results)

# ============================================================================
# COMPARE MODULE — Model Comparison & Feature Ranking
# ============================================================================

def compare_models(models, X, y, cv=5, scoring="accuracy"):
    """Compare multiple models via cross-validation.

    *models*: ``{"name": estimator, ...}``

    Returns a DataFrame with columns ``[model, mean_score, std_score]``
    sorted by ``mean_score`` descending.
    """
    rows = []
    for name, mdl in models.items():
        scores = cross_val_score(mdl, X, y, cv=cv, scoring=scoring)
        rows.append({
            "model": name,
            "mean_score": float(scores.mean()),
            "std_score": float(scores.std()),
        })
    df = pd.DataFrame(rows).sort_values("mean_score", ascending=False)
    return df.reset_index(drop=True)


def best_model(models, X, y, cv=5, scoring="accuracy"):
    """Return the ``(name, estimator)`` tuple of the best model by CV score."""
    df = compare_models(models, X, y, cv=cv, scoring=scoring)
    winner = df.iloc[0]["model"]
    return winner, models[winner]


def rank_features(model, X, y, feature_names=None):
    """Rank features by importance after fitting *model*.

    Returns a DataFrame with columns ``[feature, importance]``
    sorted by ``importance`` descending.
    """
    model.fit(X, y)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).ravel()
    else:
        raise AttributeError(
            f"{type(model).__name__} exposes neither "
            "feature_importances_ nor coef_."
        )
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(len(importances))]
    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)
