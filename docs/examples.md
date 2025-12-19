# Example Workflows

This page demonstrates how to use LongitudeML for longitudinal forecasting tasks, based on the example scripts in `examples/ptsd_stop_forecasting/`.

## 1. Forecasting with PyTorch Lightning (`run_PCL_forecast.py`)

This script shows how to set up a full training pipeline using the modules in `src/`.

### Steps:
1. **Argument Parsing**: Load default arguments for data, model, and training.
2. **Data Loading**: Use `get_datasetDict` to prepare train/val/test splits from disk or database.
3. **Logger Setup**: Initialize experiment logger with `get_logger`.
4. **DataLoader Module**: Create a `MIDataLoaderModule` for PyTorch Lightning.
5. **Model Selection**: Choose and instantiate a model (recurrent, transformer, or linear) based on arguments.
6. **Lightning Module**: Wrap the model in `MILightningModule` for training and evaluation.
7. **Training**: Use PyTorch Lightning's `Trainer` to fit the model.

### Example Code
```python
from src import get_default_args, get_logger, get_datasetDict, MIDataLoaderModule, MILightningModule, recurrent, AutoRegressiveTransformer
import pytorch_lightning as pl

args = get_default_args()
datasetDict = get_datasetDict(train_data, val_folds=args.val_folds)
logger = get_logger('comet', workspace=args.workspace, project_name=args.project_name, experiment_name=args.experiment_name, save_dir=args.output_dir)
data_module = MIDataLoaderModule(args, datasetDict)

# Model selection
if args.model_type == 'recurrent':
    model = recurrent(...)
elif args.model_type == 'transformer':
    model = AutoRegressiveTransformer(...)
# ... other model options

lightning_module = MILightningModule(args, model)
trainer = pl.Trainer(max_epochs=args.epochs, logger=logger)
trainer.fit(lightning_module, datamodule=data_module)
```

## 2. Data Preparation for Forecasting (`save_1_day_forecast_data_lang_selfreport_v6.2.py`)

This script demonstrates how to extract, merge, and preprocess longitudinal features and outcomes from a database using the data pipeline utilities.

### Steps:
1. **Initialize DLATKDataGetter**: Set up with table names and outcome fields.
2. **Extract Features and Outcomes**: Use `get_long_features` and `get_long_outcomes`.
3. **Align and Merge**: Use `intersect_seqids` and `merge_features_outcomes` to align data by sequence and time.
4. **Masking and Normalization**: Apply masking and normalization functions to handle missing data and scale features.
5. **Save Dataset**: Store the processed dataset for downstream modeling.

### Example Code
```python
from src.dlatk_datapipeline import DLATKDataGetter

getter = DLATKDataGetter(...)
long_embs, long_outcomes = getter.get_long_features(), getter.get_long_outcomes()
long_embs, long_outcomes = getter.intersect_seqids(long_embs, long_outcomes)
dataset = merge_features_outcomes(long_outcomes, long_embs)
# Further processing: masking, normalization, etc.
```

See the full scripts in `examples/ptsd_stop_forecasting/` for more details and advanced usage. 