# Evaluation Metrics

LongitudeML provides flexible evaluation metrics for regression and classification tasks, supporting sequence masking and multiple aggregation modes.

## Metric Functions

### `mi_mse`, `mi_smape`, `mi_pearsonr`, `mi_mae`
These functions compute mean squared error, symmetric mean absolute percentage error, Pearson correlation, and mean absolute error, respectively. They support sequence masking and different reduction (aggregation) modes.

### Expected Tensor Shapes
- `input` (preds): `(batch_size, seq_len, num_outcomes)`
- `target`: `(batch_size, seq_len, num_outcomes)`
- `mask`: `(batch_size, seq_len, num_outcomes)` (1=valid, 0=invalid/padded); if not provided, all elements are considered valid

## Reduction Modes (`reduction` argument)

The `reduction` argument controls how metrics are aggregated:

| Mode           | Description |
|----------------|-------------|
| `within-seq`   | Computes the metric **within each sequence** (e.g., per user or per sample), then averages across all sequences. This is the default and is useful when you want to respect the temporal or grouped structure of your data. |
| `between-seq`  | Computes the metric **between sequence means**: first averages predictions and targets over time within each sequence, then computes the metric on these means across sequences. Use this to evaluate performance at the sequence (e.g., user) level, ignoring within-sequence variation. |
| `flatten`      | Flattens all sequences and time steps into a single vector, then computes the metric globally. Use this for a global, ungrouped assessment. |
| `none`         | Returns the metric for each element (no reduction). Useful for debugging or custom aggregation. |

**When to use each mode:**
- Use `within-seq` for most longitudinal or grouped prediction tasks where you care about per-sequence accuracy.
- Use `between-seq` if your downstream task is to predict sequence-level aggregates (e.g., mean outcome per user).
- Use `flatten` for a global view, ignoring sequence boundaries.
- Use `none` for raw, unreduced metrics.

## Example Usage
```python
from src import mi_mse
# Suppose preds, targets, and mask are torch tensors of shape (batch_size, seq_len, num_outcomes)
# Example: batch_size=32, seq_len=10, num_outcomes=1
mse_within = mi_mse(input=preds, target=targets, mask=mask)  # Default: reduction='within-seq'
mse_flatten = mi_mse(input=preds, target=targets, mask=mask, reduction='flatten')
mse_between = mi_mse(input=preds, target=targets, mask=mask, reduction='between-seq')
```

See also: `examples/ptsd_stop_forecasting/run_PCL_forecast.py` for more usage examples. 