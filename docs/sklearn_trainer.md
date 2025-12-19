# Sklearn Trainer: Scikit-Learn Integration

## Overview

`SklearnModule` and `SklearnTrainer` provide a PyTorch Lightning-like interface for scikit-learn models, allowing simple sklearn models (Ridge, Lasso, etc.) to work seamlessly with PyTorch DataLoaders. This is designed for models that have `fit()` and `predict()` methods and don't require gradient-based optimization.

The sklearn trainer handles the conversion between PyTorch's 3D tensor format (batch_size, seq_len, features) and sklearn's 2D array format (n_samples, n_features), computes comprehensive evaluation metrics across multiple data subsets (OOTS/OOSS), and supports hyperparameter search via GridSearchCV and RandomizedSearchCV.

**Key Design Pattern**: Sklearn models must implement a `select_features()` method (adapter pattern) that decides which embeddings to use from the batch dictionary, mirroring how PyTorch models choose their inputs.

**Module Organization**: 
- **`sklearn_trainer.py`**: Contains `SklearnModule` and `SklearnTrainer` classes for training orchestration
- **`mi_sklearn_model.py`**: Contains sklearn model adapter classes (`RidgeForecastModel`, `LassoForecastModel`, `AutoRegressiveRidge`, `AutoRegressiveLasso`, etc.)

---

## Helper Functions

### `collect_batch_dict(dataloader)`
**What it does:**  
Aggregates all batches from a PyTorch DataLoader into a single batch dictionary by concatenating tensors along the batch dimension.

**Inputs:**  
- `dataloader`: PyTorch DataLoader yielding dictionary batches with keys like `'embeddings_*'`, `'outcomes'`, `'outcomes_mask'`, `'oots_mask'`, `'ooss_mask'`, `'time_ids'`, `'seq_id'`, etc.

**Output:**  
- `Dict[str, torch.Tensor]`: Dictionary with the same keys as individual batches, where values are concatenated tensors over all batches along dim=0.

---

### `reshape_for_sklearn(X, y, mask)`
**What it does:**  
Converts 3D PyTorch tensors to 2D numpy arrays suitable for sklearn models, filtering out invalid samples (where mask is False).

**Inputs:**  
- `X`: `(batch_size, seq_len, input_dim)` tensor
- `y`: `(batch_size, seq_len, num_outcomes)` tensor
- `mask`: `(batch_size, seq_len, num_outcomes)` boolean tensor

**Output:**  
- `X_2d`: `(n_valid_samples, input_dim)` numpy array
- `y_2d`: `(n_valid_samples, num_outcomes)` numpy array
- `valid_indices`: List of `(batch_idx, time_idx)` tuples for reconstruction

**Important Note on Multi-Outcome Models:**  
This function uses `mask.any(dim=-1)` which includes timesteps where **at least one** outcome is valid. For multi-outcome models, this means:
- If `outcomes_mask = [1, 1, 0]`, this sample **is included** in training
- Sklearn will train on all outcomes, including the invalid one (with mask=0)
- This is suboptimal for multi-outcome sklearn models as sklearn wastes effort predicting invalid outcomes

**For single-outcome models**: This is not an issue since there's only one outcome.  
**For multi-outcome models**: Consider using separate sklearn models per outcome or modifying this function to use `mask.all(dim=-1)` to only include fully valid samples.

---

### `reconstruct_from_sklearn(predictions, original_shape, valid_indices, mask)`
**What it does:**  
Reconstructs 3D predictions from sklearn's 2D output, placing predictions at valid positions and filling invalid positions with zeros.

**Inputs:**  
- `predictions`: `(n_valid_samples, num_outcomes)` numpy array
- `original_shape`: `(batch_size, seq_len, num_outcomes)` tuple
- `valid_indices`: List of `(batch_idx, time_idx)` tuples
- `mask`: Original mask tensor `(batch_size, seq_len, num_outcomes)`

**Output:**  
- `preds_3d`: `(batch_size, seq_len, num_outcomes)` tensor

---

## Method Reference

### `SklearnModule(args, model)`
**What it does:**  
Wraps a sklearn model to work with PyTorch DataLoaders, handling data reshaping, training, evaluation, and metric computation.

**Inputs:**  
- `args`: Namespace or dict containing hyperparameters and configuration (must include `metrics_reduction` for metric computation).
- `model`: Sklearn model instance with `fit()` and `predict()` methods. **Must also implement `select_features(batch_dict, args)`** that returns `(X_3d, y_3d, mask_3d)` tensors.

**Output:**  
- An object with the following key methods:
  - `fit(dataloader)`: Train the sklearn model and compute training metrics
  - `evaluate(dataloader, split='val')`: Evaluate and compute exhaustive metrics for OOTS/OOSS subsets
  - `predict(dataloader)`: Make predictions without computing metrics (no labels needed)
  - `compute_metrics(preds, targets, mask)`: Compute basic metrics (MSE, SMAPE, Pearson, MAE)
  - `compute_exhaustive_metrics(preds, targets, mask, oots_mask, ooss_mask)`: Compute metrics for multiple subsets

**What it expects as input:**  
- DataLoaders producing batch dictionaries with keys like `'embeddings_*'`, `'outcomes'`, `'outcomes_mask'`, `'oots_mask'`, `'ooss_mask'`, etc. (see [datamodule.md](datamodule.md))

**What the output looks like:**  
- Stores predictions and metrics internally in `self.predictions` and `self.metrics` dictionaries
- Returns metric dictionaries from `fit()` and `evaluate()` methods

---

### `fit(dataloader)`
**What it does:**  
Trains the sklearn model on data from the dataloader, computes training metrics, and stores predictions.

**Inputs:**  
- `dataloader`: PyTorch DataLoader

**Output:**  
- Dictionary with training metrics: `{'mse': ..., 'smape': ..., 'pearsonr': ..., 'mae': ...}`

---

### `evaluate(dataloader, split='val')`
**What it does:**  
Evaluates the sklearn model and computes metrics for multiple data subsets based on `oots_mask` and `ooss_mask`:
- `ws_wt`: within sample, within time (ooss==0 & oots==0)
- `valset`: all validation data (oots==1 OR ooss==1)
- `ws_oots`: within sample, out of time (ooss==0 & oots==1)
- `wt_ooss`: within time, out of sample (ooss==1 & oots==0)
- `oots_ooss`: out of time, out of sample (ooss==1 & oots==1)
- `oots`: all out of time samples (oots==1)
- `ooss`: all out of sample sequences (ooss==1)

**Inputs:**  
- `dataloader`: PyTorch DataLoader
- `split`: `'val'` or `'test'` (used for storing predictions/metrics)

**Output:**  
- Dictionary with evaluation metrics for all subsets: `{'ws_wt_mse': ..., 'valset_mse': ..., 'ws_oots_mse': ..., ...}`. Empty subsets return `-1.0` for all metrics.

---

### `predict(dataloader)`
**What it does:**  
Makes predictions without computing metrics (labels are optional).

**Inputs:**  
- `dataloader`: PyTorch DataLoader (may not have labels)

**Output:**  
- Dictionary with keys: `'preds'`, `'outcomes'`, `'outcomes_mask'`, `'seq_id'`, `'time_ids'`, `'oots_mask'`, `'ooss_mask'`

---

### `compute_metrics(preds, targets, mask)`
**What it does:**  
Computes evaluation metrics using `mi_eval` functions (MSE, SMAPE, Pearson correlation, MAE) with the reduction mode specified in `args.metrics_reduction`.

**Inputs:**  
- `preds`: `(batch_size, seq_len, num_outcomes)` tensor
- `targets`: `(batch_size, seq_len, num_outcomes)` tensor
- `mask`: `(batch_size, seq_len, num_outcomes)` tensor

**Output:**  
- Dictionary with metric values: `{'mse': ..., 'smape': ..., 'pearsonr': ..., 'mae': ...}`

See [evaluation.md](evaluation.md) for details on reduction modes.

---

### `compute_exhaustive_metrics(preds, targets, mask, oots_mask, ooss_mask)`
**What it does:**  
Computes metrics for multiple data subsets based on `oots_mask` and `ooss_mask`, matching the behavior of `MILightningModule.validation_step`. Handles shape validation and correctly interprets time-wise `oots_mask` and sequence-wise `ooss_mask`.

**Inputs:**  
- `preds`: `(batch_size, seq_len, num_outcomes)` tensor
- `targets`: `(batch_size, seq_len, num_outcomes)` tensor
- `mask`: `(batch_size, seq_len, num_outcomes)` tensor
- `oots_mask`: `(batch_size, seq_len)` tensor - out of time indicator (time-wise)
- `ooss_mask`: `(batch_size,)` or `(batch_size, 1)` tensor - out of sample indicator (sequence-wise)

**Output:**  
- Dictionary with metrics for all subsets (same as `evaluate()` output)

---

### `SklearnTrainer(output_dir, logger=None, **trainer_params)`
**What it does:**  
Orchestrates training, validation, testing, and hyperparameter search for sklearn models, similar to PyTorch Lightning's Trainer.

**Inputs:**  
- `output_dir`: Directory path for saving models and results
- `logger`: Optional logger instance (Comet, TensorBoard, etc.) for experiment tracking
- `**trainer_params`: Additional trainer configuration parameters

**Output:**  
- An object with the following key methods:
  - `fit(module, train_dataloader, val_dataloader=None)`: Train and optionally validate
  - `validate(module, val_dataloader)`: Validate the model
  - `test(module, test_dataloader)`: Test the model
  - `predict(module, predict_dataloader)`: Make predictions
  - `hyperparameter_search(...)`: Perform GridSearchCV or RandomizedSearchCV
  - `save_model(module, filename)`: Save module to disk
  - `load_model(filename)`: Load saved module from disk

---

### `fit(module, train_dataloader, val_dataloader=None)`
**What it does:**  
Trains the sklearn model and optionally validates it, logging metrics and saving the model.

**Inputs:**  
- `module`: `SklearnModule` instance
- `train_dataloader`: Training data loader
- `val_dataloader`: Optional validation data loader

**Output:**  
- Dictionary with keys `'train'` and `'val'`, each containing metric dictionaries

---

### `validate(module, val_dataloader)`
**What it does:**  
Validates the model (can be called anytime, not just after training).

**Inputs:**  
- `module`: `SklearnModule` instance
- `val_dataloader`: Validation data loader

**Output:**  
- Dictionary with validation metrics (exhaustive subset metrics)

---

### `test(module, test_dataloader)`
**What it does:**  
Tests the model on the test set.

**Inputs:**  
- `module`: `SklearnModule` instance
- `test_dataloader`: Test data loader

**Output:**  
- Dictionary with test metrics (exhaustive subset metrics)

---

### `predict(module, predict_dataloader)`
**What it does:**  
Makes predictions without labels.

**Inputs:**  
- `module`: `SklearnModule` instance
- `predict_dataloader`: Data loader (may not have labels)

**Output:**  
- Dictionary with predictions and metadata

---

### `hyperparameter_search(module, param_grid, train_dataloader, val_dataloader, search_type='grid', n_iter=10, cv=3)`
**What it does:**  
Performs hyperparameter search using `GridSearchCV` or `RandomizedSearchCV`, evaluates the best model on the validation set, and saves it.

**Inputs:**  
- `module`: `SklearnModule` instance (will be cloned internally)
- `param_grid`: Dictionary of parameter names to lists of values (e.g., `{'alpha': [0.1, 1.0, 10.0]}`)
- `train_dataloader`: Training data loader
- `val_dataloader`: Validation data loader
- `search_type`: `'grid'` or `'random'`
- `n_iter`: Number of iterations for random search
- `cv`: Number of cross-validation folds

**Output:**  
- `SklearnModule` instance with the best model (from `search.best_estimator_`)

---

### `save_model(module, filename)`
**What it does:**  
Saves the module (including the sklearn model, args, metrics, and predictions) to disk. Uses `output_dir/project_name/experiment_key/` structure when logger is available, otherwise saves to `output_dir/`.

**Inputs:**  
- `module`: `SklearnModule` instance
- `filename`: Name of the file to save (e.g., `'final_model.pkl'`)

**Output:**  
- None (side effect: saves to disk)

---

### `load_model(filename)`
**What it does:**  
Loads a saved module from disk, reconstructing the `SklearnModule` with the saved model, args, metrics, and predictions.

**Inputs:**  
- `filename`: Name of the file to load

**Output:**  
- `SklearnModule` instance

---

## Available Sklearn Model Classes

Pre-built sklearn model adapters are available in `mi_sklearn_model.py`:

### Basic Models (No History Windowing)
- **`RidgeForecastModel`**: Ridge regression adapter that selects embeddings from batch dictionary
- **`LassoForecastModel`**: Lasso regression adapter with the same feature selection logic

### Autoregressive Models (With History Windowing)
- **`AutoRegressiveRidge`**: Ridge regression with sliding window history (uses `args.max_len`)
- **`AutoRegressiveLasso`**: Lasso regression with sliding window history (uses `args.max_len`)

These models automatically handle history windowing when `args.max_len > 1`, expanding each timestep's features to include the current and previous `max_len - 1` timesteps.

---

## Adapter Pattern: `select_features` Interface

Sklearn models used with `SklearnModule` must implement a `select_features()` method that defines how to extract input features, targets, and masks from the aggregated batch dictionary. This mirrors the PyTorch design where the model decides which `embeddings_*` keys to use.

### Interface Signature

```python
def select_features(self, batch_dict: Dict[str, torch.Tensor], args) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args:
        batch_dict: Dictionary with keys like 'embeddings_*', 'outcomes', 'outcomes_mask', etc.
        args: Namespace with configuration
    
    Returns:
        X_3d: (batch_size, seq_len, input_dim) tensor
        y_3d: (batch_size, seq_len, num_outcomes) tensor
        mask_3d: (batch_size, seq_len, num_outcomes) boolean tensor
    """
```

### Expected Shapes
- `X_3d`: `(batch_size, seq_len, input_dim)` - Input features
- `y_3d`: `(batch_size, seq_len, num_outcomes)` - Target outcomes
- `mask_3d`: `(batch_size, seq_len, num_outcomes)` - Boolean mask indicating valid samples

### Using Pre-built Models

The easiest way to use sklearn models is to import the pre-built adapters:

```python
from src import RidgeForecastModel, AutoRegressiveRidge, SklearnModule, SklearnTrainer

# Basic Ridge model
model = RidgeForecastModel(alpha=1.0, random_state=42)

# Autoregressive Ridge with 7-day history
args.max_len = 7
model_ar = AutoRegressiveRidge(alpha=1.0, random_state=42)
```

### Custom Model Implementation

You can also create your own adapter by implementing `select_features()`:

```python
from sklearn.linear_model import Ridge

class CustomRidgeModel(Ridge):
    """Custom adapter that selects specific embeddings."""
    
    def select_features(self, batch_dict, args):
        # Your custom feature selection logic
        X_3d = batch_dict["embeddings_lang_z"]
        y_3d = batch_dict["outcomes"]
        mask_3d = batch_dict["outcomes_mask"]
        return X_3d, y_3d, mask_3d
```

---

## Key Features

- **PyTorch DataLoader Integration**: Works seamlessly with PyTorch DataLoaders, aggregating batches automatically
- **3D to 2D Conversion**: Handles reshaping between PyTorch's 3D tensor format and sklearn's 2D array format
- **Exhaustive Metric Computation**: Computes metrics for multiple OOTS/OOSS subsets, matching `MILightningModule` behavior
- **Hyperparameter Search**: Supports `GridSearchCV` and `RandomizedSearchCV` for hyperparameter tuning
- **Model Persistence**: Saves and loads models with structured paths (`output_dir/project_name/experiment_key/`)
- **Logging Integration**: Works with Comet, TensorBoard, and other loggers for experiment tracking
- **Adapter Pattern**: Flexible feature selection via `select_features()` method
- **Multi-Outcome Warning**: Warns users about potential issues with multi-outcome models

---

## Example Usage

### Basic Training and Validation

```python
from src import (
    SklearnModule, SklearnTrainer, 
    RidgeForecastModel, AutoRegressiveRidge,
    get_default_args, get_logger, get_datasetDict, MIDataLoaderModule
)
from datasets import load_from_disk

# Setup
args = get_default_args()
data = load_from_disk(args.data_dir)
datasetDict = get_datasetDict(train_data=data, val_folds=args.val_folds)
for key in datasetDict:
    datasetDict[key] = datasetDict[key].with_format('torch')

dataloaderModule = MIDataLoaderModule(args, datasetDict)
logger = get_logger('comet', workspace=args.workspace, project_name=args.project_name, 
                    experiment_name=args.experiment_name, save_dir=args.output_dir)

# Create model and module (using pre-built RidgeForecastModel)
model = RidgeForecastModel(alpha=args.weight_decay, random_state=args.seed)
module = SklearnModule(args, model)
trainer = SklearnTrainer(output_dir=args.output_dir, logger=logger)

# Train and validate
results = trainer.fit(
    module,
    train_dataloader=dataloaderModule.train_dataloader(),
    val_dataloader=dataloaderModule.val_dataloader()
)

print(f"Training metrics: {results['train']}")
print(f"Validation metrics: {results['val']}")

# Test
if dataloaderModule.test_dataloader():
    test_results = trainer.test(module, dataloaderModule.test_dataloader())
    print(f"Test metrics: {test_results}")
```

### Autoregressive Model with History Windowing

```python
from src import AutoRegressiveRidge, SklearnModule, SklearnTrainer

# Set history window size
args.max_len = 7  # Use 7-day history

# Create autoregressive model
model = AutoRegressiveRidge(alpha=args.weight_decay, random_state=args.seed)
module = SklearnModule(args, model)
trainer = SklearnTrainer(output_dir=args.output_dir, logger=logger)

# Works exactly like regular sklearn trainer!
results = trainer.fit(
    module,
    train_dataloader=dataloaderModule.train_dataloader(),
    val_dataloader=dataloaderModule.val_dataloader()
)
```

### Hyperparameter Search

```python
# Define parameter grid
param_grid = {
    'alpha': [0.1, 1.0, 10.0, 100.0]
}

# Run grid search
best_module = trainer.hyperparameter_search(
    module,
    param_grid,
    dataloaderModule.train_dataloader(),
    dataloaderModule.val_dataloader(),
    search_type='grid',
    cv=3
)

print(f"Best validation metrics: {best_module.metrics['val']}")
```

### Model Saving and Loading

```python
# Save model
trainer.save_model(module, 'final_model.pkl')

# Load model later
loaded_module = trainer.load_model('final_model.pkl')
```

See also: 
- `examples/ptsd_stop_forecasting/test_sklearn_trainer.py` for complete working examples with Ridge and Lasso models
- `src/mi_sklearn_model.py` for all available sklearn model adapter classes

