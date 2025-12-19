# LongitudeML

LongitudeML is a machine learning framework for longitudinal forecasting, designed for time-series and sequence prediction tasks in health, behavioral, and other domains. It provides utilities for data loading, preprocessing, model definition (including transformers and recurrent models), and training using PyTorch Lightning.

## Features
- Flexible data pipeline for extracting and processing longitudinal data
- Support for Huggingface Datasets and PyTorch DataLoaders
- Recurrent, transformer, and linear model architectures
- PyTorch Lightning integration for robust training and evaluation
- Flexible evaluation metrics and reduction modes for grouped/longitudinal data
- Experiment tracking and logging (CometML support)
- Modular argument/configuration system

## Installation

Clone the repository and install dependencies:

```bash
git clone <repo-url>
cd LongitudeML
pip install -r requirements.txt
```

Or use the provided `environment.yml` for a conda environment:

```bash
conda env create -f environment.yml
conda activate longitudeml
```

## Quickstart Example

```python
from src.mi_args import get_default_args
from src.mi_datamodule import get_datasetDict, MIDataLoaderModule
from src import MILightningModule, recurrent
import pytorch_lightning as pl

# Get default arguments and prepare data
data_args = get_default_args()
dataset_dict = get_datasetDict(train_data, val_data, test_data)
data_module = MIDataLoaderModule(data_args, dataset_dict)

# Initialize model and Lightning module
model = recurrent(input_size=128, hidden_size=64, num_classes=1)
lightning_module = MILightningModule(data_args, model)

# Train
trainer = pl.Trainer(max_epochs=data_args.epochs)
trainer.fit(lightning_module, datamodule=data_module)
```

See [examples.md](docs/examples.md) and the `examples/` folder for full scripts and workflows.

## Documentation

📚 **Full documentation is available at: [https://adithya8.github.io/LongitudeML](https://adithya8.github.io/LongitudeML)**

### Quick Links

- [Project Index](docs/index.md): Overview and navigation
- [Data Pipeline](docs/data_pipeline.md): Data extraction and processing
- [Data Module](docs/datamodule.md): Dataset and DataLoader setup
- [Models](docs/models.md): Model architectures and usage
- [Lightning Module](docs/lightning_module.md): Training and evaluation workflow
- [Sklearn Trainer](docs/sklearn_trainer.md): Scikit-learn model integration
- [Evaluation](docs/evaluation.md): Metrics and reduction modes
- [Logger](docs/logger.md): Experiment tracking
- [Arguments](docs/mi_args.md): All configuration options
- [Examples](docs/examples.md): End-to-end usage

## Contributing

Contributions, bug reports, and feature requests are welcome! Please open an issue or submit a pull request.

## Contact

For questions or support, please contact me or open an issue on GitHub.
