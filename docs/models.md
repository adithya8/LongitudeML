# Models: Recurrent, Transformer, and Linear Architectures

## Overview

LongitudeML provides a variety of model architectures for sequence and time-series forecasting, including recurrent neural networks (GRU), transformer-based models, and linear baselines. These are implemented in `mi_model.py` and `mi_transformers.py`.

## Key Model Classes

### `recurrent`
A GRU-based recurrent model for sequence prediction, supporting multi-instance learning and masking for variable-length sequences.

#### `forward()` Arguments
| Argument                      | Type           | Required | Description                                                                                 |
|-------------------------------|----------------|----------|---------------------------------------------------------------------------------------------|
| `embeddings`                  | torch.Tensor   | Yes      | Input tensor of shape (batch_size, seq_len, input_size)                                     |
| `hidden_rep`                  | torch.Tensor   | No       | Initial hidden state (num_layers * num_directions, batch_size, hidden_size)                 |
| `mask`                        | torch.Tensor   | No       | Boolean mask (batch_size, seq_len); 1=valid, 0=invalid                                      |
| `predict_last_valid_hidden_state` | bool        | No       | If True, predict only for last valid timestep; else predict for all timesteps (default=True) |
| `**kwargs`                    | dict           | No       | Additional arguments (not typically used)                                                    |

### `AutoRegressiveTransformer`
A transformer model for autoregressive sequence modeling, with support for custom positional encodings and multi-head attention.

#### `forward()` Arguments
| Argument                      | Type           | Required | Description                                                                                 |
|-------------------------------|----------------|----------|---------------------------------------------------------------------------------------------|
| `embeddings`                  | torch.Tensor   | Yes      | Input tensor of shape (batch_size, seq_len, input_size)                                     |
| `mask`                        | torch.Tensor   | No       | Boolean mask (batch_size, seq_len); 1=padded, 0=valid                                       |
| `predict_last_valid_hidden_state` | bool        | No       | If True, predict only for last valid timestep; else predict for all timesteps (default=True) |
| `**kwargs` (e.g., `time_ids`) | dict           | No       | Additional arguments, e.g., `time_ids` for masking                                          |

### `TransformerModel`
A flexible transformer implementation supporting sinusoidal, rotary, or no positional encoding, and configurable for different input/output dimensions.

#### `forward()` Arguments
| Argument                      | Type           | Required | Description                                                                                 |
|-------------------------------|----------------|----------|---------------------------------------------------------------------------------------------|
| `embeddings`                  | torch.Tensor   | Yes      | Input tensor of shape (batch_size, seq_len, input_dim)                                      |
| `mask`                        | torch.Tensor   | No       | Attention mask (batch_size, 1, seq_len, seq_len); -inf for masked, 0 for unmasked           |
| `**kwargs` (e.g., `time_ids`) | dict           | No       | Additional arguments, e.g., `time_ids` for masking                                          |

### `AutoRegressiveLinear`
Linear model using causal convolution for sequence prediction.

#### `forward()` Arguments
| Argument                      | Type           | Required | Description                                                                                 |
|-------------------------------|----------------|----------|---------------------------------------------------------------------------------------------|
| `embeddings`                  | torch.Tensor   | Yes      | Input tensor of shape (batch_size, seq_len, input_size)                                     |
| `mask`                        | torch.Tensor   | Yes      | Boolean mask (batch_size, seq_len); 1=padded, 0=valid                                       |
| `predict_last_valid_hidden_state` | bool        | No       | If True, predict only for last valid timestep; else predict for all timesteps (default=True) |
| `**kwargs`                    | dict           | No       | Additional arguments                                                                        |

### `AutoRegressiveLinear2`
Linear model using sliding window for autoregressive prediction.

#### `forward()` Arguments
| Argument                      | Type           | Required | Description                                                                                 |
|-------------------------------|----------------|----------|---------------------------------------------------------------------------------------------|
| `embeddings`                  | torch.Tensor   | Yes      | Input tensor of shape (batch_size, seq_len, input_size)                                     |
| `mask`                        | torch.Tensor   | No       | Boolean mask (batch_size, seq_len); 1=padded, 0=valid                                       |
| `predict_last_valid_hidden_state` | bool        | No       | If True, predict only for last valid timestep; else predict for all timesteps (default=True) |
| `**kwargs`                    | dict           | No       | Additional arguments                                                                        |

### `BoELinear`
Bag-of-Embeddings linear model, averages embeddings before linear projection.

#### `forward()` Arguments
| Argument                      | Type           | Required | Description                                                                                 |
|-------------------------------|----------------|----------|---------------------------------------------------------------------------------------------|
| `embeddings`                  | torch.Tensor   | Yes      | Input tensor of shape (batch_size, seq_len, input_size)                                     |
| `mask`                        | torch.Tensor   | No       | Boolean mask (batch_size, seq_len); 1=padded, 0=valid                                       |
| `predict_last_valid_hidden_state` | bool        | No       | If True, predict only for last valid timestep; else predict for all timesteps (default=True) |
| `**kwargs`                    | dict           | No       | Additional arguments                                                                        |

## Example Usage
```python
from src import recurrent, AutoRegressiveTransformer, TransformerModel

# Recurrent model
model = recurrent(input_size=128, hidden_size=64, num_classes=1, num_layers=2)
output = model(embeddings, mask=mask)

# Transformer model
tr_model = AutoRegressiveTransformer(
    input_size=128, hidden_size=64, num_classes=1, num_layers=2, num_heads=4, max_len=100
)
output = tr_model(embeddings, mask=mask, time_ids=time_ids)

# Linear baseline
from src import AutoRegressiveLinear
lin_model = AutoRegressiveLinear(input_size=128, hidden_size=64, num_classes=1)
output = lin_model(embeddings, mask=mask)
```

See also: `examples/ptsd_stop_forecasting/run_PCL_forecast.py` for model instantiation and selection logic. 