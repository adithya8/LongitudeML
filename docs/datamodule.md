# Data Module: Dataset Preparation and Loading

## Overview

The data module provides utilities for converting raw data dictionaries into Huggingface Datasets, creating masks for missing data, and preparing PyTorch DataLoaders for model training and evaluation.

## Method Reference

### `get_dataset(data)`
**What it does:**  
Converts a dictionary of data (with keys like `'embeddings'`, `'labels'`, etc.) into a Huggingface `Dataset` object. Automatically creates a `'time_ids'` field if not present.

**Inputs:**  
- `data` (dict): Dictionary with keys such as `'embeddings'`, `'labels'`, and optionally `'time_ids'`.

**Output:**  
- Huggingface `Dataset` object, with all fields as columns.

---

### `get_datasetDict(train_data, val_data=None, test_data=None, val_folds=None, test_folds=None, fold_col='folds')`
**What it does:**  
Creates a `DatasetDict` with 'train', 'val', and 'test' splits from input dictionaries or Datasets. Handles fold-based splitting and adds out-of-sample (`ooss_mask`) indicators.

**Inputs:**  
- `train_data`, `val_data`, `test_data` (dict or Dataset): Data for each split.
- `val_folds`, `test_folds` (list, optional): Fold indices for validation/test splits.
- `fold_col` (str): Name of the fold column.

**Output:**  
- Huggingface `DatasetDict` with keys `'train'`, `'val'`, `'test'` (as available), each a `Dataset` with added `'ooss_mask'` column.

---

### `create_mask(examples)`
**What it does:**  
Adds mask fields to each sequence for handling missing time points and infilling missing vectors/labels. Used with `.map()` on a Dataset.

**Inputs:**  
- `examples` (dict): A single example or batch from a Dataset, with `'time_ids'`, `'embeddings'`, `'labels'`, etc.

**Output:**  
- Modified example(s) with added `'mask'` and `'infill_mask'` fields, indicating valid/missing time points.

---

### `default_collate_fn(features, predict_last_valid_timestep, partition)`
**What it does:**  
Custom collate function for PyTorch DataLoader, handling padding, masking, and batching of variable-length sequences.

**Inputs:**  
- `features` (list of dict): List of examples from the Dataset.
- `predict_last_valid_timestep` (bool): Whether to predict only the last valid timestep.
- `partition` (str): One of `'train'`, `'val'`, or `'test'`.

**Output:**  
- Batch dictionary with all fields padded and batched appropriately for model input.

---

### `MIDataLoaderModule(data_args, datasets, partion_processing=True)`
**What it does:**  
A PyTorch Lightning `LightningDataModule` that wraps the above utilities and provides `train_dataloader`, `val_dataloader`, and `test_dataloader` methods.

**Inputs:**  
- `data_args`: Namespace or object with data-related arguments (e.g., batch size, predict_last_valid_timestep).
- `datasets` (dict): Dictionary with keys `'train'`, `'val'`, `'test'` (each a Dataset).
- `partion_processing` (bool): Whether to use partition-specific collate functions.

**Output:**  
- An object with methods:
  - `train_dataloader()`: Returns a DataLoader for training.
  - `val_dataloader()`: Returns a DataLoader for validation.
  - `test_dataloader()`: Returns a DataLoader for testing.

---

## Example Usage
```python
from src.mi_datamodule import get_datasetDict, MIDataLoaderModule

# Prepare datasets
dataset_dict = get_datasetDict(train_data, val_data, test_data)

# Create DataLoader module for PyTorch Lightning
data_module = MIDataLoaderModule(args, dataset_dict)
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()
```

See also: `examples/ptsd_stop_forecasting/run_PCL_forecast.py` for how these are used in practice. 