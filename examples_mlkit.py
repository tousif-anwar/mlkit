"""
MLKit - One-Liner Examples
Examples showing how to use MLKit for common ML/DL/CV tasks
"""

import numpy as np
import pandas as pd
import mlkit
from pathlib import Path


print("""
╔═══════════════════════════════════════════════════════════════════╗
║                    MLKIT - ONE-LINER EXAMPLES                     ║
║            Like lenskit, but for ML/DL/Computer Vision            ║
╚═══════════════════════════════════════════════════════════════════╝
""")

# ============================================================================
# EXAMPLE 1: QUICK EVALUATION
# ============================================================================

print("\n[1] QUICK EVALUATION - One Liner")
print("="*60)

# Generate sample data
y_true = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 0])
y_pred = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 0])

# ONE-LINER: Get complete metrics report
metrics = mlkit.eval.evaluate(y_true, y_pred)
print(f"Metrics: {metrics}")

# ONE-LINER: Print formatted report
mlkit.eval.report(y_true, y_pred)


# ============================================================================
# EXAMPLE 2: COMPARE MULTIPLE MODELS
# ============================================================================

print("\n[2] COMPARE MODELS - One Liner")
print("="*60)

# Sample results from different models
results = {
    'LogisticRegression': {'accuracy': 0.92, 'precision': 0.91, 'recall': 0.90, 'f1': 0.90},
    'RandomForest': {'accuracy': 0.95, 'precision': 0.94, 'recall': 0.94, 'f1': 0.94},
    'SVM': {'accuracy': 0.88, 'precision': 0.87, 'recall': 0.88, 'f1': 0.87},
    'NeuralNet': {'accuracy': 0.96, 'precision': 0.96, 'recall': 0.96, 'f1': 0.96}
}

# ONE-LINER: Compare models
comparison_df = mlkit.eval.compare(results)
print(comparison_df)

# ONE-LINER: Visualize comparison
mlkit.viz.compare_models(results, metric='accuracy', show=True)


# ============================================================================
# EXAMPLE 3: VISUALIZE CONFUSION MATRIX
# ============================================================================

print("\n[3] CONFUSION MATRIX - One Liner")
print("="*60)

# Generate multi-class data
y_true_multi = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
y_pred_multi = np.array([0, 1, 1, 0, 1, 2, 0, 2, 2, 0, 1, 2])

# ONE-LINER: Plot confusion matrix
mlkit.viz.confusion(y_true_multi, y_pred_multi, 
                   labels=['Class_A', 'Class_B', 'Class_C'], 
                   show=True)


# ============================================================================
# EXAMPLE 4: PLOT TRAINING HISTORY
# ============================================================================

print("\n[4] TRAINING HISTORY - One Liner")
print("="*60)

# Simulate training history
epochs = 50
history = {
    'loss': np.logspace(0, -1, epochs) + np.random.normal(0, 0.01, epochs),
    'val_loss': np.logspace(0, -1, epochs) + np.random.normal(0, 0.015, epochs),
    'accuracy': np.linspace(0.5, 0.95, epochs) + np.random.normal(0, 0.01, epochs),
    'val_accuracy': np.linspace(0.5, 0.93, epochs) + np.random.normal(0, 0.015, epochs)
}

# ONE-LINER: Plot all training curves
mlkit.viz.history(history, show=True)

# ONE-LINER: Plot specific metrics only
mlkit.viz.history(history, metrics=['loss', 'val_loss'], show=True)


# ============================================================================
# EXAMPLE 5: LEARNING CURVES (OVERFITTING DETECTION)
# ============================================================================

print("\n[5] LEARNING CURVES - One Liner")
print("="*60)

train_scores = np.linspace(0.5, 0.95, 30) + np.random.normal(0, 0.02, 30)
val_scores = np.linspace(0.5, 0.92, 30) + np.random.normal(0, 0.03, 30)

# ONE-LINER: Plot to detect overfitting
mlkit.viz.learning_curve(train_scores, val_scores, show=True)


# ============================================================================
# EXAMPLE 6: ROC CURVE
# ============================================================================

print("\n[6] ROC CURVE - One Liner")
print("="*60)

y_true_binary = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 0] * 5)
y_pred_proba = np.random.rand(len(y_true_binary))

# ONE-LINER: Plot ROC curve with AUC
mlkit.viz.roc(y_true_binary, y_pred_proba, show=True)


# ============================================================================
# EXAMPLE 7: DATA ANALYSIS
# ============================================================================

print("\n[7] DATA ANALYSIS - One Liner")
print("="*60)

# Generate sample dataset
X = np.random.randn(1000, 10)
y = np.random.randint(0, 3, 1000)

# ONE-LINER: Analyze data
analysis = mlkit.data.analyze(X)
print(f"Analysis: {analysis}")

# ONE-LINER: Print summary
mlkit.data.summary(X)

# ONE-LINER: Split data
splits = mlkit.data.split(X, y, test_size=0.2, val_size=0.1)
print(f"Splits: {splits['splits']}")

# ONE-LINER: Normalize data
X_norm, params = mlkit.data.normalize(X, method='zscore')
print(f"Data normalized with params: {list(params.keys())}")


# ============================================================================
# EXAMPLE 8: DISTRIBUTION ANALYSIS
# ============================================================================

print("\n[8] DISTRIBUTIONS - One Liner")
print("="*60)

# ONE-LINER: Plot distributions
mlkit.viz.distributions(X[:, :5], show=True)


# ============================================================================
# EXAMPLE 9: TRUE VS PREDICTED
# ============================================================================

print("\n[9] PREDICTIONS SCATTER - One Liner")
print("="*60)

y_true_reg = np.linspace(0, 10, 50)
y_pred_reg = y_true_reg + np.random.normal(0, 1, 50)

# ONE-LINER: Plot predictions
mlkit.viz.predictions(y_true_reg, y_pred_reg, show=True)


# ============================================================================
# EXAMPLE 10: COMPUTER VISION - IMAGE BATCH
# ============================================================================

print("\n[10] COMPUTER VISION - View Batch")
print("="*60)

# Generate sample images
batch = np.random.randint(0, 256, (9, 28, 28, 3), dtype=np.uint8)
labels = ['cat', 'dog', 'bird', 'cat', 'dog', 'bird', 'cat', 'dog', 'bird']

# ONE-LINER: View image batch
mlkit.vision.view_batch(batch, labels=labels, grid_size=(3, 3), show=True)


# ============================================================================
# EXAMPLE 11: COMPUTER VISION - SALIENCY MAP
# ============================================================================

print("\n[11] COMPUTER VISION - Saliency Map")
print("="*60)

# Generate sample image
image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)

# ONE-LINER: Display saliency map
mlkit.vision.saliency_map(image, show=True)


# ============================================================================
# EXAMPLE 12: TRAINING TRACKING
# ============================================================================

print("\n[12] TRAINING TRACKING - One Liner")
print("="*60)

# Simulate training loop
print("Simulating training:")
for epoch in range(5):
    metrics = {
        'loss': 1.0 - epoch * 0.15,
        'acc': 0.5 + epoch * 0.08
    }
    # ONE-LINER: Track metrics
    mlkit.train.track(epoch, metrics)


# ============================================================================
# EXAMPLE 13: EARLY STOPPING
# ============================================================================

print("\n[13] EARLY STOPPING - One Liner")
print("="*60)

val_losses = [1.0, 0.9, 0.85, 0.84, 0.84, 0.84, 0.85, 0.86]

# ONE-LINER: Check if should stop
for epoch, loss in enumerate(val_losses):
    should_stop = mlkit.train.early_stop(val_losses[:epoch+1], patience=3, mode='min')
    print(f"Epoch {epoch}: loss={loss:.4f}, should_stop={should_stop}")


# ============================================================================
# EXAMPLE 14: LEARNING RATE SCHEDULING
# ============================================================================

print("\n[14] LEARNING RATE SCHEDULING - One Liner")
print("="*60)

# ONE-LINER: Get LR for each epoch
print("Learning rates by epoch:")
for epoch in range(0, 100, 10):
    lr = mlkit.train.lr_schedule(epoch, initial_lr=0.001, schedule_type='exponential')
    print(f"  Epoch {epoch:3d}: lr = {lr:.6f}")


# ============================================================================
# EXAMPLE 15: SAVE AND LOAD RESULTS
# ============================================================================

print("\n[15] SAVE & LOAD RESULTS - One Liner")
print("="*60)

results_data = {
    'model': 'RandomForest',
    'metrics': {'accuracy': 0.95, 'f1': 0.94},
    'hyperparams': {'n_estimators': 100, 'max_depth': 10},
    'training_time': 45.2
}

# ONE-LINER: Save results
mlkit.results.save(results_data, './experiment_results.json', format='json')

# ONE-LINER: Load results
loaded = mlkit.results.load('./experiment_results.json')
print(f"Loaded results: {loaded['model']}")

# ONE-LINER: Print summary
mlkit.results.summary(results_data)


# ============================================================================
# EXAMPLE 16: COMPARE HYPERPARAMETERS
# ============================================================================

print("\n[16] HYPERPARAMETER COMPARISON - One Liner")
print("="*60)

hp_results = {
    'lr_0.001': {'accuracy': 0.92, 'f1': 0.91},
    'lr_0.01': {'accuracy': 0.95, 'f1': 0.94},
    'lr_0.1': {'accuracy': 0.88, 'f1': 0.87},
    'lr_0.001_decay': {'accuracy': 0.96, 'f1': 0.96}
}

# ONE-LINER: Compare hyperparameters
mlkit.compare.hyperparams(hp_results, metric='accuracy', show=True)


# ============================================================================
# EXAMPLE 17: COMPARE DATASETS
# ============================================================================

print("\n[17] COMPARE DATASETS - One Liner")
print("="*60)

dataset_results = {
    'MNIST': {'accuracy': 0.99, 'training_time': 120},
    'CIFAR10': {'accuracy': 0.95, 'training_time': 450},
    'ImageNet': {'accuracy': 0.92, 'training_time': 2400}
}

# ONE-LINER: Compare datasets
dataset_comparison = mlkit.compare.datasets(dataset_results)
print(dataset_comparison)


# ============================================================================
# EXAMPLE 18: REPRODUCIBILITY
# ============================================================================

print("\n[18] REPRODUCIBILITY - One Liner")
print("="*60)

# ONE-LINER: Set seed for reproducibility
mlkit.seed(42)
print("Seed set to 42")

# Generate reproducible random data
data1 = np.random.randn(5)
mlkit.seed(42)  # Reset seed
data2 = np.random.randn(5)
print(f"Data reproducibility: {np.allclose(data1, data2)}")


# ============================================================================
# EXAMPLE 19: ENVIRONMENT INFO
# ============================================================================

print("\n[19] ENVIRONMENT INFO - One Liner")
print("="*60)

# ONE-LINER: Get environment info
mlkit.print_info()


# ============================================================================
# EXAMPLE 20: COMPLETE WORKFLOW
# ============================================================================

print("\n[20] COMPLETE WORKFLOW - All One-Liners")
print("="*60)

print("Step 1: Generate data")
X = np.random.randn(1000, 20)
y = np.random.randint(0, 2, 1000)

print("Step 2: Analyze data")
mlkit.data.summary(X)

print("Step 3: Split data")
splits = mlkit.data.split(X, y)

print("Step 4: Normalize")
X_train_norm, params = mlkit.data.normalize(splits['X_train'])

print("Step 5: Evaluate predictions")
y_pred = np.random.randint(0, 2, len(y))
metrics = mlkit.eval.evaluate(y, y_pred)
mlkit.eval.report(y, y_pred)

print("Step 6: Visualize results")
mlkit.viz.confusion(y, y_pred)

print("Step 7: Save results")
mlkit.results.save({'metrics': metrics, 'splits': splits['splits']}, 'workflow_results.json')

print("\nWorkflow complete!")


print("""
╔═══════════════════════════════════════════════════════════════════╗
║                    ALL EXAMPLES COMPLETED!                        ║
║                                                                   ║
║  MLKit provides simple one-liner functions for:                   ║
║  • Evaluation (eval.evaluate, eval.compare)                       ║
║  • Visualization (viz.confusion, viz.history, viz.roc)            ║
║  • Data analysis (data.analyze, data.split, data.normalize)       ║
║  • Computer vision (vision.view_batch, vision.saliency_map)      ║
║  • Training utilities (train.track, train.early_stop, train.lr)   ║
║  • Results management (results.save, results.load)                ║
║  • Comparisons (compare.models, compare.hyperparams)              ║
║                                                                   ║
║  Import and use: import mlkit                                     ║
║                  mlkit.eval.evaluate(y_true, y_pred)              ║
║                  mlkit.viz.confusion(y_true, y_pred)              ║
╚═══════════════════════════════════════════════════════════════════╝
""")
