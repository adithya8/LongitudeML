# Argument Configuration: mi_args.py

## Overview

The `mi_args.py` module provides functions to add and parse all command-line arguments and configuration options for data, model, training, and logging. These are typically used to build an `argparse.ArgumentParser` and produce a Namespace object for use throughout the pipeline.

## Key Functions and Arguments

### `get_data_args(parser)`
Adds arguments related to data input/output (e.g., data directory, file paths, output directory) to the parser.

| Argument                | Type    | Default   | Description                                 |
|------------------------|---------|-----------|---------------------------------------------|
| `--data_dir`           | str     | None      | Path to data directory                      |
| `--data_file`          | str     | None      | Path to data file                           |
| `--train_file`         | str     | None      | Path to training data file                  |
| `--val_file`           | str     | None      | Path to validation data file                |
| `--test_file`          | str     | None      | Path to test data file                      |
| `--output_dir`         | str     | None      | Output directory                            |
| `--overwrite_output_dir`| flag   | False     | Overwrite output directory                  |

---

### `get_model_args(parser)`
Adds arguments related to model architecture and hyperparameters.

| Argument                    | Type    | Default      | Description                                                      |
|----------------------------|---------|--------------|------------------------------------------------------------------|
| `--model_type`              | str     | 'gru'        | Model type (gru, trns, custom, baseline, custom_scratch)         |
| `--custom_model`            | str     | None         | Custom model name                                                |
| `--input_size`              | int     | 768          | Size of the embeddings                                           |
| `--num_classes`             | int     | 1            | Number of classes                                                |
| `--num_outcomes`            | int     | 1            | Number of outcomes                                               |
| `--hidden_size`             | int     | 128          | Hidden size                                                      |
| `--projection_size`         | int     | None         | Projection size (default: None, same as hidden_size)              |
| `--num_layers`              | int     | 1            | Number of layers                                                 |
| `--dropout`                 | float   | 0.10         | Dropout rate                                                     |
| `--output_dropout`          | float   | 0.10         | Output layer dropout                                             |
| `--positional_encoding_type`| str     | 'none'       | Positional encoding type (none, sinusoidal, learned, rope)        |
| `--pre_ln`                  | flag    | False        | Use pre-layer normalization                                      |
| `--bidirectional`           | flag    | False        | Use bidirectional RNN                                            |
| `--num_heads`               | int     | 2            | Number of heads for transformer model                            |
| `--max_len`                 | int     | 130          | Maximum sequence length for transformer model                    |
| `--max_history_len`         | int     | None         | Maximum history length for transformer model                     |
| `--sliding_window_size`     | int     | None         | Sliding window size for recalibration                            |

---

### `get_training_args(parser)`
Adds arguments related to training, loss/metric reduction, early stopping, and more.

| Argument                        | Type    | Default      | Description                                                      |
|----------------------------------|---------|--------------|------------------------------------------------------------------|
| `--do_train`                     | flag    | False        | Run training                                                     |
| `--do_test`                      | flag    | False        | Run testing                                                      |
| `--val_folds`                    | list    | []           | Folds to validate on                                             |
| `--do_hparam_tune`               | flag    | False        | Run hyperparameter tuning                                        |
| `--n_trials`                     | int     | 100          | Number of trials for tuning                                      |
| `--min_epochs`                   | int     | 1            | Minimum number of epochs                                         |
| `--epochs`                       | int     | None         | Number of epochs to train                                        |
| `--train_batch_size`             | int     | 32           | Batch size for training                                          |
| `--val_batch_size`               | int     | 64           | Batch size for evaluation                                        |
| `--cross_entropy_class_weight`   | list    | None         | Class weights for cross entropy loss                             |
| `--loss_reduction`               | str     | 'flatten'    | Loss reduction strategy (within-seq, flatten, none)              |
| `--metrics_reduction`            | str     | 'within-seq' | Metrics reduction strategy (within-seq, between-seq, flatten, none)|
| `--do_shift`                     | flag    | False        | Predict change instead of absolute value                         |
| `--interpolated_output`          | flag    | False        | Output is linearly interpolated                                  |
| `--seq_len_scheduler_type`       | str     | 'none'       | Sequence length scheduler type (none, linear, exponential)       |
| `--min_seq_len`                  | int     | 10           | Minimum sequence length for training                             |
| `--max_seq_len`                  | int     | -1           | Maximum sequence length for training                             |
| `--max_scheduled_epochs`         | int     | -1           | Max epochs for sequence length scheduler                         |
| `--log_interval`                 | int     | 10           | Logging interval                                                 |
| `--save_strategy`                | str     | 'best'       | Model save strategy (best, all)                                  |
| `--save_dir`                     | str     | None         | Model save directory                                             |
| `--lr`                           | float   | 0.001        | Learning rate                                                    |
| `--weight_decay`                 | float   | 0.0          | Weight decay                                                     |
| `--lr_scheduler`                 | str     | 'none'       | Learning rate scheduler (none, linear)                           |
| `--warmup_epochs`                | int     | 1            | Number of warmup epochs for scheduler                            |
| `--start_factor`                 | float   | 0.5          | Start factor for scheduler                                       |
| `--num_workers`                  | int     | 4            | Number of workers                                                |
| `--seed`                         | int     | 42           | Random seed                                                      |
| `--predict_last_valid_timestep`  | flag    | False        | Predict from last valid timestep only                            |
| `--early_stopping_patience`      | int     | 0            | Patience for early stopping                                      |
| `--early_stopping_min_delta`     | float   | 0.0          | Min delta for early stopping                                     |
| `--early_stopping_mode`          | str     | 'min'        | Early stopping mode (min, max)                                   |
| `--subscale_weights_path`        | str     | None         | Path to subscale model weights                                   |
| `--lang_weights_path`            | str     | None         | Path to language model weights                                   |

---

### `get_comet_args(parser)`
Adds arguments for CometML experiment tracking.

| Argument                | Type    | Default         | Description                                 |
|------------------------|---------|-----------------|---------------------------------------------|
| `--api_key`            | str     | ~/.comet.key    | CometML API key                             |
| `--workspace`          | str     | None            | CometML workspace                           |
| `--project_name`       | str     | None            | CometML project name                        |
| `--experiment_name`    | str     | None            | CometML experiment name                     |

---

### `get_default_args(jupyter=False)`
Returns a Namespace object with all default arguments for data, model, training, and logging, ready to use in scripts.

**Inputs:**
- `jupyter` (bool): If True, adds Jupyter-specific arguments for notebook compatibility.

**Output:**
- `argparse.Namespace` object with all arguments and their default values.

---

## Example Usage

```python
from src.mi_args import get_default_args

args = get_default_args()
print(args.data_dir, args.model_type, args.epochs, args.workspace, ...)
``` 