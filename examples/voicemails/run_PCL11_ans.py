import os, pickle, shutil
from datetime import datetime
from utils import add_to_path
add_to_path(__file__)

from numpy import min as np_min
import pytorch_lightning as pl
import optuna
from optuna.pruners import HyperbandPruner
from optuna.integration import PyTorchLightningPruningCallback

from src import (
    get_default_args, get_logger,
    get_datasetDict, create_mask, MIDataLoaderModule,
    MILightningModule
)


def objective(trial: optuna.trial.Trial, args, dataloaderModule):
    # Set the seed for reproducibility. Model init with same weight for every trial
    pl.seed_everything(args.seed)
    
    args.lr = trial.suggest_float("lr", 9e-5, 1e-2, log=True)
    args.weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)
    args.output_dropout = trial.suggest_float("output_dropout", 0.0, 0.5)
    args.hidden_size = trial.suggest_categorical("hidden_size", [16, 32, 64])
    
    lightning_module = MILightningModule(args)

    trial_number = trial.number
    trial_exp_name = args.experiment_name+'_trial_{}'.format(trial_number)

    # Get the logger and log the hyperparameters
    logger = get_logger('comet', workspace=args.workspace, project_name=args.project_name, experiment_name=trial_exp_name, save_dir=args.output_dir)
    logger.log_hyperparams(args.__dict__)
    
    # add callback for early stopping
    callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss")]
    if args.early_stopping_patience>0: callbacks=[pl.callbacks.EarlyStopping(monitor="val_loss", patience=args.early_stopping_patience, 
                                                                             mode=args.early_stopping_mode, min_delta=args.early_stopping_min_delta)]
    trainer = pl.Trainer(accelerator='gpu', devices=1, default_root_dir=args.output_dir, logger=logger,
                        callbacks=callbacks, min_epochs=args.min_epochs, max_epochs=args.epochs, enable_checkpointing=False)
    trainer.fit(lightning_module, train_dataloaders=dataloaderModule.train_dataloader(), val_dataloaders=dataloaderModule.val_dataloader())
    
    score = np_min(lightning_module.epoch_loss['val'])
    logger.experiment.end()
    
    SRC_PATH = os.path.join(args.output_dir, '{}/{}/'.format(logger._project_name, logger._experiment_key))
    DEST_PATH = args.HPARAM_OUTPUT_DIR + '/{}/'.format(logger._experiment_key)
    print ('Moving {} to {}'.format(SRC_PATH, DEST_PATH))
    if os.path.exists(SRC_PATH): shutil.move(SRC_PATH, DEST_PATH)
    
    return score


if __name__ == '__main__':
    # Get the args from command line
    args = get_default_args()
    
    # Get the datasetDict
    dataDict = {'train_data': None, 'val_data': None, 'test_data': None}

    if args.data_dir is not None:
        for split in ['train', 'val', 'test']:
            file_path = os.path.join(args.data_dir, '{}.pkl'.format(split))
            if os.path.exists(file_path):
                dataDict['{}_data'.format(split)] = pickle.load(open(file_path, 'rb'))
    elif args.data_file is not None:
        if os.path.exists(args.data_file):
            dataDict = pickle.load(open(args.data_file, 'rb'))
        else:
            raise Warning('data_file:{} does not exist.'.format(args.data_file))
    #TODO: Check logic for elif
    elif (args.train_file is not None) or (args.val_file is not None) or (args.test_file is not None):
        if args.train_file is None and (args.do_train or args.do_nfold_cv):
            raise Warning('train_file not provided. Training will not be performed.')
        else:
            dataDict['train_data'] = pickle.load(open(args.train_file, 'rb'))
        if args.val_file is None:
            # TODO: Add validation split
            raise Warning('val_file not provided. Validation split will be performed.')
        else:
            dataDict['val_data'] = pickle.load(open(args.val_file, 'rb'))
        if args.test_file is None and args.do_test:
            raise Warning('test_file not provided. Testing will not be performed.')
        else:
            dataDict['test_data'] = pickle.load(open(args.test_file, 'rb'))
    else:
        raise Warning('data_dir or train_file, val_file, test_file must be provided.')

    if ('train_data' not in dataDict or not dataDict['train_data']) and (args.do_train or args.do_nfold_cv):
        print ('train_data not provided. Training will not be performed.')
        dataDict['train_data'] = None
    if ('val_data' not in dataDict or not dataDict['val_data']):
        print ('val_data not provided. Validation split will be performed.')
        dataDict['val_data'] = None
    if ('test_data' not in dataDict or not dataDict['test_data']):
        if args.do_test: print ('test_data not provided. Testing will not be performed.')
        dataDict['test_data'] = None
    
    # Get the datasetDict (in HF datasetDict format) containing the embeddings across the temporal dimension for each sequence along with the labels and sequence numbers
    datasetDict = get_datasetDict(train_data=dataDict['train_data'], val_data=dataDict['val_data'], test_data=dataDict['test_data'], val_folds=args.val_folds)
    
    # Create a mask pattern for the sequence
    datasetDict = datasetDict.map(create_mask)

    if not os.path.exists(args.output_dir):
        print ('Creating output directory: {}'.format(args.output_dir))
        os.makedirs(args.output_dir)
        
    if args.do_hparam_tune:
        # Get the dataloader module
        date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        args.HPARAM_OUTPUT_DIR = os.path.join(args.output_dir, args.project_name, 'hparam_'+date_time)
        os.makedirs(args.HPARAM_OUTPUT_DIR)
        
        dataloaderModule = MIDataLoaderModule(args, datasetDict)
        
        sampler = optuna.samplers.TPESampler(seed=args.seed)
        study = optuna.create_study(direction="minimize", sampler=sampler, pruner=HyperbandPruner(max_resource=args.epochs))
        study.optimize(lambda trial: objective(trial, args, dataloaderModule), n_trials=args.n_trials)

        print("Number of finished trials: {}".format(len(study.trials)))
        print("Best trial:")
        best_trial = study.best_trial
        
        # Print value and trial number
        print ("  Value: {}".format(best_trial.value))
        print ("  Number: {}".format(best_trial.number))
        
        print ("  Params: ")
        for key, value in best_trial.params.items():
            print ("    {}: {}".format(key, value))
        
        # Save trials to output_dir as csv
        study.trials_dataframe().to_csv(os.path.join(args.HPARAM_OUTPUT_DIR, 'trials.csv'))
        
    elif args.do_train:
        # Set the seed for reproducibility
        pl.seed_everything(args.seed)
        
        # Get the logger and log the hyperparameters
        logger = get_logger('comet', workspace=args.workspace, project_name=args.project_name, experiment_name=args.experiment_name, save_dir=args.output_dir)
        logger.log_hyperparams(args.__dict__)
        
        # Get the dataloader module
        dataloaderModule = MIDataLoaderModule(args, datasetDict)
        
        if not os.path.exists(args.output_dir):
            print ('Creating output directory: {}'.format(args.output_dir))
            os.makedirs(args.output_dir)
        
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
        # if args.do_test
