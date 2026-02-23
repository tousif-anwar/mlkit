"""Tests for mlkit.py — Framework-Agnostic ML Toolkit."""

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier

import mlkit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def binary_data():
    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    return X, y


@pytest.fixture
def regression_data():
    X, y = make_regression(n_samples=200, n_features=5, random_state=42)
    return X, y


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "a": [1.0, 2.0, np.nan, 4.0, 5.0],
        "b": [10.0, np.nan, 30.0, 40.0, 50.0],
        "c": ["x", "y", "x", "z", "y"],
    })


# ============================================================================
# EVAL MODULE
# ============================================================================

class TestEval:
    def test_accuracy(self):
        assert mlkit.accuracy([1, 0, 1], [1, 0, 1]) == 1.0

    def test_precision(self):
        val = mlkit.precision([1, 0, 1, 0], [1, 0, 0, 0])
        assert 0.0 <= val <= 1.0

    def test_recall(self):
        val = mlkit.recall([1, 0, 1, 0], [1, 0, 0, 0])
        assert 0.0 <= val <= 1.0

    def test_f1(self):
        val = mlkit.f1([1, 0, 1, 0], [1, 0, 0, 0])
        assert 0.0 <= val <= 1.0

    def test_roc_auc(self):
        val = mlkit.roc_auc([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2])
        assert 0.0 <= val <= 1.0

    def test_confusion_mat(self):
        cm = mlkit.confusion_mat([1, 0, 1, 0], [1, 0, 0, 0])
        assert cm.shape == (2, 2)

    def test_class_report(self):
        rpt = mlkit.class_report([1, 0, 1, 0], [1, 0, 0, 0])
        assert isinstance(rpt, str)
        assert "precision" in rpt

    def test_mse(self):
        val = mlkit.mse([1, 2, 3], [1, 2, 3])
        assert val == 0.0

    def test_rmse(self):
        val = mlkit.rmse([1, 2, 3], [1, 2, 3])
        assert val == 0.0

    def test_mae(self):
        val = mlkit.mae([1, 2, 3], [1.5, 2.5, 3.5])
        assert val == pytest.approx(0.5)

    def test_r2(self):
        val = mlkit.r2([1, 2, 3], [1, 2, 3])
        assert val == pytest.approx(1.0)

    def test_log_loss_score(self):
        val = mlkit.log_loss_score([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2])
        assert val > 0.0


# ============================================================================
# VIZ MODULE
# ============================================================================

class TestViz:
    def test_plot_confusion_matrix(self):
        fig = mlkit.plot_confusion_matrix([1, 0, 1, 0], [1, 0, 0, 0])
        assert fig is not None

    def test_plot_roc_curve(self):
        fig = mlkit.plot_roc_curve([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2])
        assert fig is not None

    def test_plot_precision_recall(self):
        fig = mlkit.plot_precision_recall([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2])
        assert fig is not None

    def test_plot_feature_importance(self):
        fig = mlkit.plot_feature_importance([0.3, 0.1, 0.6], ["a", "b", "c"])
        assert fig is not None

    def test_plot_feature_importance_from_model(self, binary_data):
        X, y = binary_data
        model = RandomForestClassifier(n_estimators=5, random_state=42).fit(X, y)
        fig = mlkit.plot_feature_importance(model)
        assert fig is not None

    def test_plot_correlation_matrix(self):
        df = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
        fig = mlkit.plot_correlation_matrix(df)
        assert fig is not None

    def test_plot_distribution(self):
        fig = mlkit.plot_distribution(np.random.randn(100))
        assert fig is not None

    def test_plot_distribution_column(self):
        df = pd.DataFrame({"val": np.random.randn(100)})
        fig = mlkit.plot_distribution(df, column="val")
        assert fig is not None

    def test_plot_save(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            mlkit.plot_confusion_matrix([1, 0], [1, 0], save_path=path)
            assert os.path.exists(path)
        finally:
            os.unlink(path)


# ============================================================================
# DATA MODULE
# ============================================================================

class TestData:
    def test_split(self, binary_data):
        X, y = binary_data
        X_tr, X_te, y_tr, y_te = mlkit.split(X, y, test_size=0.3)
        assert len(X_tr) + len(X_te) == len(X)

    def test_normalize(self):
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        transformed, scaler = mlkit.normalize(X)
        assert transformed.min() == pytest.approx(0.0)
        assert transformed.max() == pytest.approx(1.0)

    def test_standardize(self):
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        transformed, scaler = mlkit.standardize(X)
        assert transformed.mean(axis=0) == pytest.approx([0.0, 0.0], abs=1e-7)

    def test_fill_missing_mean(self, sample_df):
        filled = mlkit.fill_missing(sample_df[["a", "b"]], strategy="mean")
        assert filled.isna().sum().sum() == 0

    def test_fill_missing_median(self, sample_df):
        filled = mlkit.fill_missing(sample_df[["a", "b"]], strategy="median")
        assert filled.isna().sum().sum() == 0

    def test_fill_missing_mode(self, sample_df):
        filled = mlkit.fill_missing(sample_df[["c"]], strategy="mode")
        assert filled.isna().sum().sum() == 0

    def test_fill_missing_scalar(self, sample_df):
        filled = mlkit.fill_missing(sample_df[["a", "b"]], strategy=0)
        assert filled.isna().sum().sum() == 0

    def test_encode_labels(self):
        encoded, le = mlkit.encode_labels(["cat", "dog", "cat", "bird"])
        assert set(encoded) == {0, 1, 2}

    def test_one_hot_encode(self, sample_df):
        encoded = mlkit.one_hot_encode(sample_df)
        assert "c" not in encoded.columns
        assert encoded.shape[1] > sample_df.shape[1]

    def test_remove_outliers_iqr(self):
        df = pd.DataFrame({"v": [1, 2, 3, 4, 5, 100]})
        cleaned = mlkit.remove_outliers(df, "v")
        assert len(cleaned) < len(df)

    def test_remove_outliers_zscore(self):
        df = pd.DataFrame({"v": [1, 2, 3, 4, 5, 100]})
        cleaned = mlkit.remove_outliers(df, "v", method="zscore", threshold=2)
        assert len(cleaned) <= len(df)

    def test_balance_classes(self):
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([0, 0, 0, 1, 1])
        X_b, y_b = mlkit.balance_classes(X, y)
        _, counts = np.unique(y_b, return_counts=True)
        assert counts[0] == counts[1]

    def test_reduce_dims(self, binary_data):
        X, _ = binary_data
        reduced, pca = mlkit.reduce_dims(X, n_components=2)
        assert reduced.shape == (200, 2)


# ============================================================================
# VISION MODULE
# ============================================================================

class TestVision:
    def _make_img(self, size=(32, 32, 3)):
        return np.random.randint(0, 256, size, dtype=np.uint8)

    def test_load_and_resize(self):
        from PIL import Image
        img = self._make_img((64, 64, 3))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            Image.fromarray(img).save(f.name)
            path = f.name
        try:
            loaded = mlkit.load_image(path)
            assert loaded.shape == (64, 64, 3)
            resized = mlkit.resize_image(loaded, (32, 32))
            assert resized.shape == (32, 32, 3)
        finally:
            os.unlink(path)

    def test_augment_image(self):
        img = self._make_img()
        aug = mlkit.augment_image(img, flip_horizontal=True, brightness_delta=10)
        assert aug.shape == img.shape
        assert aug.dtype == np.uint8

    def test_to_tensor(self):
        img = self._make_img()  # (32, 32, 3)
        t = mlkit.to_tensor(img)
        assert t.shape == (3, 32, 32)
        assert t.dtype == np.float32
        assert 0.0 <= t.min() and t.max() <= 1.0

    def test_show_image(self):
        img = self._make_img()
        fig = mlkit.show_image(img, title="test")
        assert fig is not None


# ============================================================================
# TRAIN MODULE
# ============================================================================

class TestTrain:
    def test_cross_val(self, binary_data):
        X, y = binary_data
        scores = mlkit.cross_val(LogisticRegression(max_iter=200), X, y)
        assert len(scores) == 5

    def test_grid_search(self, binary_data):
        X, y = binary_data
        gs = mlkit.grid_search(
            DecisionTreeClassifier(),
            {"max_depth": [2, 5]},
            X, y,
        )
        assert hasattr(gs, "best_params_")

    def test_train_and_predict(self, binary_data):
        X, y = binary_data
        model = mlkit.train_model(LogisticRegression(max_iter=200), X, y)
        preds = mlkit.predict(model, X)
        assert len(preds) == len(y)

    def test_predict_proba(self, binary_data):
        X, y = binary_data
        model = mlkit.train_model(LogisticRegression(max_iter=200), X, y)
        proba = mlkit.predict_proba(model, X)
        assert proba.shape[0] == len(y)

    def test_save_and_load(self, binary_data):
        X, y = binary_data
        model = LogisticRegression(max_iter=200).fit(X, y)
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            path = f.name
        try:
            mlkit.save_model(model, path)
            loaded = mlkit.load_model(path)
            np.testing.assert_array_equal(model.predict(X), loaded.predict(X))
        finally:
            os.unlink(path)


# ============================================================================
# RESULTS MODULE
# ============================================================================

class TestResults:
    def test_summarize_classification(self):
        s = mlkit.summarize([1, 0, 1, 0], [1, 0, 0, 0])
        assert set(s.keys()) == {"accuracy", "precision", "recall", "f1"}

    def test_summarize_regression(self):
        s = mlkit.summarize([1.0, 2.0, 3.0], [1.1, 2.1, 3.1], task="regression")
        assert set(s.keys()) == {"mse", "rmse", "mae", "r2"}

    def test_log_results_json(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        os.unlink(path)  # ensure file does not exist
        try:
            mlkit.log_results({"acc": 0.95}, path, fmt="json")
            with open(path) as f:
                data = json.loads(f.readline())
            assert data["acc"] == 0.95
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_log_results_csv(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        os.unlink(path)
        try:
            mlkit.log_results({"acc": 0.95, "loss": 0.05}, path, fmt="csv")
            df = pd.read_csv(path)
            assert len(df) == 1
            assert df.iloc[0]["acc"] == pytest.approx(0.95)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_to_dataframe(self):
        df = mlkit.to_dataframe({"a": 1, "b": 2})
        assert len(df) == 1
        assert list(df.columns) == ["a", "b"]

    def test_to_dataframe_list(self):
        df = mlkit.to_dataframe([{"a": 1}, {"a": 2}])
        assert len(df) == 2


# ============================================================================
# COMPARE MODULE
# ============================================================================

class TestCompare:
    def test_compare_models(self, binary_data):
        X, y = binary_data
        models = {
            "lr": LogisticRegression(max_iter=200),
            "dt": DecisionTreeClassifier(random_state=42),
        }
        df = mlkit.compare_models(models, X, y)
        assert len(df) == 2
        assert "mean_score" in df.columns

    def test_best_model(self, binary_data):
        X, y = binary_data
        models = {
            "lr": LogisticRegression(max_iter=200),
            "dt": DecisionTreeClassifier(random_state=42),
        }
        name, est = mlkit.best_model(models, X, y)
        assert name in models

    def test_rank_features(self, binary_data):
        X, y = binary_data
        df = mlkit.rank_features(DecisionTreeClassifier(random_state=42), X, y)
        assert len(df) == 10
        assert "importance" in df.columns
        # Check sorted descending
        assert df["importance"].is_monotonic_decreasing


# ============================================================================
# HELPER
# ============================================================================

class TestHelper:
    def test_to_numpy_list(self):
        result = mlkit._to_numpy([1, 2, 3])
        assert isinstance(result, np.ndarray)

    def test_to_numpy_series(self):
        result = mlkit._to_numpy(pd.Series([1, 2, 3]))
        assert isinstance(result, np.ndarray)
