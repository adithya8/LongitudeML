# Logging Utilities

LongitudeML provides logging utilities for experiment tracking, hyperparameter logging, and result saving, with support for CometML and other backends.

## Logger Function

### `get_logger`
Initializes a logger for experiment tracking. Supports logging hyperparameters, metrics, and saving results to disk or remote services (e.g., CometML).

#### Example Usage
```python
from src import get_logger
logger = get_logger('comet', workspace='my_workspace', project_name='my_project', experiment_name='exp1', save_dir='./logs')
logger.log_hyperparams(args.__dict__)
```

- `workspace`, `project_name`, `experiment_name`, and `save_dir` are typical arguments for configuring the logger.
- The logger can be passed to PyTorch Lightning's Trainer for integrated experiment tracking.

See also: `examples/ptsd_stop_forecasting/run_PCL_forecast.py` for logger usage in a full training pipeline. 