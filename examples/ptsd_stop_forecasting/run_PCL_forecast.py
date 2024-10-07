import os, pickle, shutil
from datetime import datetime
from utils import add_to_path
add_to_path(__file__)

from numpy import min as np_min
import pytorch_lightning as pl
# import optuna
# from optuna.pruners import HyperbandPruner
# from optuna.integration import PyTorchLightningPruningCallback

from datasets import load_from_disk

from src import (
    get_default_args, get_logger,
    get_datasetDict, create_mask, MIDataLoaderModule,
    MILightningModule
)


if __name__ == '__main__':
    
    args = get_default_args()
    data = load_from_disk(args.data_dir)

    datasetDict = get_datasetDict(train_data=data, val_folds=[0])

    # Set the seed for reproducibility
    pl.seed_everything(args.seed)
    
    # Get the logger and log the hyperparameters
    logger = get_logger('comet', workspace=args.workspace, project_name=args.project_name, experiment_name=args.experiment_name, save_dir=args.output_dir)
    logger.log_hyperparams(args.__dict__)

    if not os.path.exists(args.output_dir):
        print ('Creating output directory: {}'.format(args.output_dir))
        os.makedirs(args.output_dir)

    # TODO: collate function should be handled by the task
    dataloaderModule = MIDataLoaderModule(args, datasetDict)

    callbacks=[pl.callbacks.EarlyStopping(monitor="val_loss", patience=args.early_stopping_patience, 
                                        mode=args.early_stopping_mode, min_delta=args.early_stopping_min_delta)] if args.early_stopping_patience>0 else []
    callbacks.append(pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, save_last=False))

    trainer = pl.Trainer(accelerator='gpu', devices=1, default_root_dir=args.output_dir, logger=logger,
                        callbacks=callbacks, min_epochs=args.min_epochs, max_epochs=args.epochs)
        
    lightning_module = MILightningModule(args) 
    
    trainer.fit(lightning_module, train_dataloaders=dataloaderModule.train_dataloader(), val_dataloaders=dataloaderModule.val_dataloader())

    # Save predictions to output_dir    
    with open(os.path.join(args.output_dir, '{}/{}/preds.pkl'.format(logger._project_name, logger._experiment_key)), 'wb') as f:
        pickle.dump(lightning_module.labels, f, protocol=pickle.HIGHEST_PROTOCOL)
    print ('Saved predictions to {}'.format(os.path.join(args.output_dir, '{}/{}/preds.pkl'.format(logger._project_name, logger._experiment_key))))

    # Save epoch metrics to output_dir
    with open(os.path.join(args.output_dir, '{}/{}/epoch_metrics.pkl'.format(logger._project_name, logger._experiment_key)), 'wb') as f:
        pickle.dump(lightning_module.epoch_metrics, f, protocol=pickle.HIGHEST_PROTOCOL)
    print ('Saved epoch metrics to {}'.format(os.path.join(args.output_dir, '{}/{}/epoch_metrics.pkl'.format(logger._project_name, logger._experiment_key))))
        
    logger.experiment.end()    