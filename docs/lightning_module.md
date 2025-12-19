# Lightning Module: Training and Evaluation

## Overview

`MILightningModule` is a PyTorch Lightning module that wraps a model and handles training, validation, and testing logic, including loss computation, metrics, and optimizer configuration.

## Method Reference

### `MILightningModule(args, model)`
**What it does:**  
Wraps a model in a PyTorch Lightning module, handling training, validation, and testing logic, including loss computation, metrics, optimizer configuration, and logging.

**Inputs:**  
- `args`: Namespace or object containing all hyperparameters and configuration options (see `mi_args.py`).
- `model`: A PyTorch model instance (recurrent, transformer, linear, etc.).

**Output:**  
- An object with the following key methods (all PyTorch Lightning conventions):
  - `training_step(batch, batch_idx)`: Runs a training step, computes loss and metrics, logs results.
  - `validation_step(batch, batch_idx)`: Runs a validation step, computes loss and metrics, logs results.
  - `test_step(batch, batch_idx)`: Runs a test step, computes loss and metrics, logs results.
  - `configure_optimizers()`: Returns optimizer (and optionally scheduler) for training.
  - `predict_step(batch, batch_idx)`: (Optional) Runs a prediction step.
  - `on_train_epoch_end`, `on_validation_epoch_end`, `on_test_epoch_end`: (Optional) Custom logic at the end of each epoch.

**What it expects as input:**  
- Batches as produced by the DataLoader (see `datamodule.md`), with keys like `'embeddings'`, `'outcomes'`, `'mask'`, etc.

**What the output looks like:**  
- Returns loss and logs metrics for each step/epoch.  
- Stores predictions, targets, and metrics for later analysis.

---

## Step-by-Step Breakdown

The `MILightningModule` follows these main steps during training, validation, and testing:

### 1. Unpack Batch Inputs
**What it is:**
- Extracts model inputs, labels, and sequence IDs from the batch dictionary.

**Expected Input:**
- Batch dictionary from the DataLoader, with keys like `'embeddings'`, `'outcomes'`, `'mask'`, `'query_ids'`, `'seq_id'`, etc.

**Output:**
- Model input dictionary, labels tensor, and sequence IDs.

---

### 2. Label Processing (Pre-Forward)
**What it is:**
- Optionally shifts or interpolates labels for change prediction or time-shifted tasks, preparing them for loss computation.

**Expected Input:**
- Labels tensor, batch metadata, and configuration flags (e.g., `do_shift`, `interpolated_output`).

**Output:**
- Processed labels tensor (may be differenced or interpolated).

---

### 3. Forward Pass
**What it is:**
- Passes the input batch through the model to obtain predictions.

**Expected Input:**
- Model input dictionary (e.g., `'embeddings'`, `'mask'`, etc.)

**Output:**
- Model output tensor (predictions for each sequence/timestep).

---

### 4. Label Processing (Post-Forward)
**What it is:**
- Optionally "reshifts" or post-processes model outputs to align with the original label space (e.g., undoing differencing or interpolation for evaluation/metrics).

**Expected Input:**
- Model output tensor, original labels tensor, mask tensor, and configuration flags.

**Output:**
- Adjusted model output tensor, aligned with the original label space.

---

### 5. Loss Computation
**What it is:**
- Computes the loss between model predictions and (processed) labels, using the specified loss function and reduction mode.

**Expected Input:**
- Model output tensor, labels tensor, mask tensor, and loss configuration (e.g., reduction mode).

**Output:**
- Scalar loss value (or tensor, if no reduction).

---

### 6. Metrics Calculation
**What it is:**
- Computes evaluation metrics (e.g., MSE, SMAPE, Pearson, MAE) for the batch, using the specified reduction mode.

**Expected Input:**
- Model output tensor, labels tensor, mask tensor, and metrics configuration.

**Output:**
- Dictionary of metric values (e.g., `{'mse': ..., 'smape': ...}`)

---

### 7. Logging and Output Storage
**What it is:**
- Logs loss and metrics to the logger, and stores predictions, labels, and metrics for later analysis.

**Expected Input:**
- Loss value, metrics dictionary, batch metadata, and logger instance.

**Output:**
- None (side effect: logs and stores results internally)

---

### 8. Optimizer Step (Training Only)
**What it is:**
- Performs an optimizer step to update model parameters (if in training mode).

**Expected Input:**
- Computed loss value, optimizer instance.

**Output:**
- None (side effect: updates model parameters)

---

## Key Features
- Integrates any LongitudeML model (recurrent, transformer, linear, etc.)
- Supports custom loss functions and metrics (MSE, SMAPE, Pearson correlation, MAE)
- Handles sequence masking, out-of-sample indicators, and time-shifted labels
- Configures optimizers and learning rate schedulers
- Stores and logs predictions, losses, and metrics for each epoch

## Example Usage
```python
from src import MILightningModule

# Assume `args` is a namespace of hyperparameters and `model` is a model instance
lightning_module = MILightningModule(args, model)

# Use with PyTorch Lightning Trainer
import pytorch_lightning as pl
trainer = pl.Trainer(max_epochs=args.epochs, logger=logger)
trainer.fit(lightning_module, datamodule=data_module)
```

See also: `examples/ptsd_stop_forecasting/run_PCL_forecast.py` for a full training pipeline. 