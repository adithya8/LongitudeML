# LongitudeML Documentation

LongitudeML is a machine learning framework for longitudinal forecasting, particularly suited for time-series and sequence prediction tasks in health and behavioral domains. It provides utilities for data loading, preprocessing, model definition (including transformers and recurrent models), and training using PyTorch Lightning.

## Documentation Structure

- **[index.md](index.md)**: Project overview and documentation guide (this file).
- **[data_pipeline.md](data_pipeline.md)**: Data extraction, processing, and preparation pipeline, with detailed method reference for `DLATKDataGetter` and related utilities.
- **[datamodule.md](datamodule.md)**: Dataset preparation, masking, and DataLoader setup, with detailed method reference for all key functions and classes.
- **[models.md](models.md)**: Model architectures (recurrent, transformer, linear), including argument tables for `forward()` methods.
- **[lightning_module.md](lightning_module.md)**: PyTorch Lightning integration for training and evaluation, with a step-by-step breakdown of the workflow.
- **[evaluation.md](evaluation.md)**: Evaluation metrics, reduction modes, expected tensor shapes, and usage examples.
- **[logger.md](logger.md)**: Logging utilities for experiment tracking and hyperparameter logging.
- **[mi_args.md](mi_args.md)**: Full argument/configuration reference, with tables for all command-line/configuration options.
- **[utils.md](utils.md)**: Quick links to evaluation and logger documentation.
- **[examples.md](examples.md)**: Walkthroughs of example scripts, showing how to use the package end-to-end.

## Main Components

- **Data Pipeline**: Utilities for extracting and processing longitudinal data from databases ([data_pipeline.md](data_pipeline.md)).
- **Data Module**: Classes and functions for preparing datasets and dataloaders ([datamodule.md](datamodule.md)).
- **Models**: Recurrent, transformer, and linear models for sequence prediction ([models.md](models.md)).
- **Lightning Module**: PyTorch Lightning integration for training and evaluation ([lightning_module.md](lightning_module.md)).
- **Evaluation**: Flexible metrics and reduction modes for grouped/longitudinal data ([evaluation.md](evaluation.md)).
- **Logging**: Experiment tracking and hyperparameter logging ([logger.md](logger.md)).
- **Arguments**: All configuration options and command-line arguments ([mi_args.md](mi_args.md)).
- **Examples**: End-to-end workflows and script walkthroughs ([examples.md](examples.md)).

## Example Usage

See [examples.md](examples.md) for walkthroughs of typical workflows using LongitudeML. 