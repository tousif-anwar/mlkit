"""
MLKit PyTorch - One-Liner Examples
Complete examples showing PyTorch integration
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import mlkit_torch as mlkit
from pathlib import Path

print("""
╔═══════════════════════════════════════════════════════════════════╗
║              MLKIT PYTORCH - ONE-LINER EXAMPLES                   ║
║           Deep Learning with Simple, Elegant APIs                 ║
╚═══════════════════════════════════════════════════════════════════╝
""")

# ============================================================================
# EXAMPLE 1: DEVICE MANAGEMENT
# ============================================================================

print("\n[1] DEVICE MANAGEMENT - One Liner")
print("="*60)

# ONE-LINER: Get device
dev = mlkit.device.get()
print(f"Using device: {dev}")

# ONE-LINER: Print device info
mlkit.device.print_info()


# ============================================================================
# EXAMPLE 2: DATA PREPARATION
# ============================================================================

print("\n[2] DATA PREPARATION - One Liner")
print("="*60)

# Generate sample data
np.random.seed(42)
torch.manual_seed(42)

X = np.random.randn(1000, 20).astype(np.float32)
y = np.random.randint(0, 2, 1000)

# ONE-LINER: Analyze data
mlkit.data.summary(X)

# ONE-LINER: Split data
splits = mlkit.data.split(X, y, test_size=0.2, val_size=0.1)
print(f"Data splits: {splits['splits']}")

# ONE-LINER: Normalize data
X_train_norm, params = mlkit.data.normalize(splits['X_train'], method='zscore')
print(f"Normalized data shape: {X_train_norm.shape}")

# ONE-LINER: Convert to tensor
X_train_tensor = mlkit.data.to_tensor(X_train_norm)
y_train_tensor = mlkit.data.to_tensor(splits['y_train'])
print(f"Tensor shapes: X={X_train_tensor.shape}, y={y_train_tensor.shape}")

# ONE-LINER: Create DataLoader
train_loader = mlkit.data.loader(X_train_tensor, y_train_tensor, batch_size=32)
val_loader = mlkit.data.loader(mlkit.data.to_tensor(splits['X_val']), 
                               mlkit.data.to_tensor(splits['y_val']), 
                               batch_size=32, shuffle=False)
print(f"DataLoader created with {len(train_loader)} batches")


# ============================================================================
# EXAMPLE 3: MODEL SETUP
# ============================================================================

print("\n[3] MODEL SETUP - One Liner")
print("="*60)

# Define simple model
class SimpleNet(nn.Module):
    def __init__(self, input_size=20):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SimpleNet().to(dev)

# ONE-LINER: Count parameters
param_count = mlkit.models.count_params(model)
print(f"Model parameters: {param_count}")

# ONE-LINER: Freeze/unfreeze layers
mlkit.models.freeze(model, freeze=False)
print("Model layers unfrozen for training")


# ============================================================================
# EXAMPLE 4: TRAINING STEP
# ============================================================================

print("\n[4] TRAINING STEP - One Liner")
print("="*60)

optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Generate single batch
x_batch = torch.randn(32, 20).to(dev)
y_batch = torch.randint(0, 2, (32,)).to(dev)

# ONE-LINER: Single training step
loss = mlkit.torch_train.step(model, x_batch, y_batch, optimizer, loss_fn, device=dev)
print(f"Training step loss: {loss:.4f}")


# ============================================================================
# EXAMPLE 5: TRAINING EPOCH
# ============================================================================

print("\n[5] TRAINING EPOCH - One Liner")
print("="*60)

# ONE-LINER: Train for one epoch
epoch_loss = mlkit.torch_train.batch(model, train_loader, optimizer, loss_fn, device=dev)
print(f"Epoch loss: {epoch_loss:.4f}")

# ONE-LINER: Validate
val_metrics = mlkit.torch_train.validate(model, val_loader, loss_fn, device=dev)
print(f"Validation loss: {val_metrics['val_loss']:.4f}")


# ============================================================================
# EXAMPLE 6: COMPLETE TRAINING LOOP
# ============================================================================

print("\n[6] COMPLETE TRAINING LOOP - One Liner")
print("="*60)

# Reset model
model = SimpleNet().to(dev)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ONE-LINER: Full training
print("Training for 20 epochs...")
history = mlkit.torch_train.fit(
    model, train_loader, val_loader, optimizer, loss_fn,
    epochs=20, device=dev, patience=5
)

print(f"Training complete! Final train loss: {history['train_loss'][-1]:.4f}")
print(f"Final val loss: {history['val_loss'][-1]:.4f}")


# ============================================================================
# EXAMPLE 7: PREDICTIONS
# ============================================================================

print("\n[7] PREDICTIONS - One Liner")
print("="*60)

# ONE-LINER: Get predictions
y_pred, y_true = mlkit.torch_train.predict(model, val_loader, device=dev)
print(f"Predictions shape: {y_pred.shape}")
print(f"Ground truth shape: {y_true.shape}")


# ============================================================================
# EXAMPLE 8: EVALUATION
# ============================================================================

print("\n[8] EVALUATION - One Liner")
print("="*60)

# ONE-LINER: Evaluate
metrics = mlkit.eval.evaluate(y_true, y_pred)
print(f"Metrics: {metrics}")

# ONE-LINER: Print report
mlkit.eval.report(y_true, y_pred)


# ============================================================================
# EXAMPLE 9: VISUALIZE TRAINING HISTORY
# ============================================================================

print("\n[9] TRAINING HISTORY - One Liner")
print("="*60)

# ONE-LINER: Plot training curves
mlkit.viz.history(history, show=True)

# ONE-LINER: Plot learning curve
train_scores = history['train_loss']
val_scores = history['val_loss']
mlkit.viz.learning_curve(train_scores, val_scores, show=True)


# ============================================================================
# EXAMPLE 10: CONFUSION MATRIX
# ============================================================================

print("\n[10] CONFUSION MATRIX - One Liner")
print("="*60)

# ONE-LINER: Plot confusion matrix
mlkit.viz.confusion(y_true, np.argmax(y_pred, axis=1),
                   labels=['Class 0', 'Class 1'], show=True)


# ============================================================================
# EXAMPLE 11: MODEL SAVE/LOAD
# ============================================================================

print("\n[11] MODEL SAVE/LOAD - One Liner")
print("="*60)

# ONE-LINER: Save model
mlkit.models.save(model, 'trained_model.pt')

# ONE-LINER: Load model
model_loaded = SimpleNet().to(dev)
model_loaded = mlkit.models.load(model_loaded, 'trained_model.pt', device=dev)
print("Model loaded successfully")


# ============================================================================
# EXAMPLE 12: LEARNING RATE SCHEDULING
# ============================================================================

print("\n[12] LEARNING RATE SCHEDULING - One Liner")
print("="*60)

print("Learning rates by epoch:")
for epoch in [0, 5, 10, 15, 20]:
    lr = mlkit.torch_train.lr_schedule(epoch, 0.001, schedule_type='exponential')
    print(f"  Epoch {epoch:2d}: {lr:.6f}")

# ONE-LINER: Update LR
new_lr = mlkit.torch_train.lr_schedule(10, 0.001)
mlkit.torch_train.update_lr(optimizer, new_lr)
print(f"Learning rate updated to {new_lr:.6f}")


# ============================================================================
# EXAMPLE 13: COMPUTER VISION - IMAGE BATCH
# ============================================================================

print("\n[13] COMPUTER VISION - Image Batch")
print("="*60)

# Generate sample images
image_batch = np.random.randint(0, 256, (9, 28, 28, 3), dtype=np.uint8) / 255.0
labels = ['cat', 'dog', 'bird', 'cat', 'dog', 'bird', 'cat', 'dog', 'bird']

# ONE-LINER: View batch
mlkit.vision.view_batch(image_batch, labels=labels, grid_size=(3, 3), show=True)


# ============================================================================
# EXAMPLE 14: COMPUTER VISION - SALIENCY MAP
# ============================================================================

print("\n[14] COMPUTER VISION - Saliency Map")
print("="*60)

# Generate sample image
image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8) / 255.0

# ONE-LINER: Visualize saliency
mlkit.vision.saliency_map(image, show=True)


# ============================================================================
# EXAMPLE 15: COMPUTER VISION - FEATURE MAPS
# ============================================================================

print("\n[15] COMPUTER VISION - Feature Maps")
print("="*60)

# Generate sample feature maps
feature_maps = torch.randn(1, 16, 32, 32)

# ONE-LINER: Visualize feature maps
mlkit.vision.feature_maps(feature_maps, n_maps=12, show=True)


# ============================================================================
# EXAMPLE 16: COMPARE MODELS
# ============================================================================

print("\n[16] COMPARE MODELS - One Liner")
print("="*60)

# Create results from different models
results = {
    'SimpleNet': {'accuracy': 0.88, 'precision': 0.87, 'f1': 0.87},
    'ResNet18': {'accuracy': 0.92, 'precision': 0.91, 'f1': 0.91},
    'DenseNet': {'accuracy': 0.89, 'precision': 0.88, 'f1': 0.88},
    'EfficientNet': {'accuracy': 0.94, 'precision': 0.93, 'f1': 0.93}
}

# ONE-LINER: Compare models
comparison = mlkit.compare.models(results)
print(comparison)

# ONE-LINER: Visualize comparison
mlkit.viz.compare_models(results, show=True)


# ============================================================================
# EXAMPLE 17: HYPERPARAMETER COMPARISON
# ============================================================================

print("\n[17] HYPERPARAMETER COMPARISON - One Liner")
print("="*60)

hp_results = {
    'lr_0.0001': {'accuracy': 0.85, 'f1': 0.84},
    'lr_0.001': {'accuracy': 0.88, 'f1': 0.87},
    'lr_0.01': {'accuracy': 0.82, 'f1': 0.81},
    'lr_0.001_wd_0.0001': {'accuracy': 0.91, 'f1': 0.90}
}

# ONE-LINER: Compare hyperparameters
mlkit.compare.hyperparams(hp_results, metric='accuracy', show=True)


# ============================================================================
# EXAMPLE 18: SAVE AND LOAD RESULTS
# ============================================================================

print("\n[18] SAVE & LOAD RESULTS - One Liner")
print("="*60)

experiment_data = {
    'model': 'SimpleNet',
    'metrics': metrics,
    'hyperparams': {'lr': 0.001, 'batch_size': 32, 'epochs': 20},
    'history': history,
    'device': str(dev)
}

# ONE-LINER: Save results
mlkit.results.save(experiment_data, './pytorch_results.json')

# ONE-LINER: Load results
loaded_data = mlkit.results.load('./pytorch_results.json')
print(f"Loaded results for model: {loaded_data['model']}")

# ONE-LINER: Print summary
mlkit.results.summary(experiment_data)


# ============================================================================
# EXAMPLE 19: REPRODUCIBILITY
# ============================================================================

print("\n[19] REPRODUCIBILITY - One Liner")
print("="*60)

# ONE-LINER: Set seed
mlkit.seed(42)
print("Seed set to 42 for reproducibility")

# Generate reproducible tensors
t1 = torch.randn(3, 3)
mlkit.seed(42)
t2 = torch.randn(3, 3)
print(f"Tensors are identical: {torch.allclose(t1, t2)}")


# ============================================================================
# EXAMPLE 20: ENVIRONMENT INFO
# ============================================================================

print("\n[20] ENVIRONMENT INFO - One Liner")
print("="*60)

# ONE-LINER: Get environment info
mlkit.print_info()


# ============================================================================
# EXAMPLE 21: DATA VISUALIZATION
# ============================================================================

print("\n[21] DATA VISUALIZATION - One Liner")
print("="*60)

# ONE-LINER: Plot distributions
mlkit.viz.distributions(X[:, :5], show=True)

# ONE-LINER: Plot predictions
y_pred_test = torch.randn(100, 2)
y_true_test = torch.randint(0, 2, (100,))
mlkit.viz.confusion(y_true_test, torch.argmax(y_pred_test, dim=1), show=True)


# ============================================================================
# EXAMPLE 22: COMPLETE DEEP LEARNING WORKFLOW
# ============================================================================

print("\n[22] COMPLETE WORKFLOW - All One-Liners")
print("="*60)

print("Step 1: Setup")
mlkit.seed(42)
mlkit.device.print_info()

print("\nStep 2: Prepare Data")
X_raw = np.random.randn(500, 15).astype(np.float32)
y_raw = np.random.randint(0, 2, 500)
data_splits = mlkit.data.split(X_raw, y_raw)
mlkit.data.summary(X_raw)

print("\nStep 3: Create Model")
model = SimpleNet(input_size=15).to(dev)
params = mlkit.models.count_params(model)
print(f"Model has {params['trainable']} trainable parameters")

print("\nStep 4: Prepare Loaders")
X_train_norm, norm_params = mlkit.data.normalize(data_splits['X_train'])
train_tensor = mlkit.data.to_tensor(X_train_norm, data_splits['y_train'])
train_loader = mlkit.data.loader(train_tensor, batch_size=32)

val_tensor = mlkit.data.to_tensor(data_splits['X_val'], data_splits['y_val'])
val_loader = mlkit.data.loader(val_tensor, batch_size=32, shuffle=False)

print("\nStep 5: Train")
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
history = mlkit.torch_train.fit(model, train_loader, val_loader, 
                               optimizer, loss_fn, epochs=15, device=dev)

print("\nStep 6: Evaluate")
y_pred, y_true = mlkit.torch_train.predict(model, val_loader, device=dev)
metrics = mlkit.eval.evaluate(y_true, y_pred)
mlkit.eval.report(y_true, y_pred)

print("\nStep 7: Visualize")
mlkit.viz.history(history, show=True)
mlkit.viz.confusion(y_true, np.argmax(y_pred, axis=1), show=True)

print("\nStep 8: Save")
mlkit.models.save(model, 'final_model.pt')
mlkit.results.save({
    'metrics': metrics,
    'history': history,
    'params': params
}, 'workflow_results.json')

print("\n✅ Complete workflow finished!")


print("""
╔═══════════════════════════════════════════════════════════════════╗
║                    ALL EXAMPLES COMPLETED!                        ║
║                                                                   ║
║  MLKit PyTorch provides simple one-liner functions for:           ║
║  • Device management (device.get, device.info)                    ║
║  • Data handling (data.split, data.normalize, data.loader)        ║
║  • Training (torch_train.step, torch_train.fit)                   ║
║  • Evaluation (eval.evaluate, eval.report)                        ║
║  • Visualization (viz.history, viz.confusion, viz.roc)            ║
║  • Computer vision (vision.view_batch, vision.feature_maps)       ║
║  • Model utilities (models.count_params, models.save)             ║
║  • Results management (results.save, results.load)                ║
║                                                                   ║
║  Import and use: import mlkit_torch as mlkit                      ║
║                  model = SimpleNet()                              ║
║                  history = mlkit.torch_train.fit(...)             ║
║                  mlkit.viz.history(history)                       ║
╚═══════════════════════════════════════════════════════════════════╝
""")
