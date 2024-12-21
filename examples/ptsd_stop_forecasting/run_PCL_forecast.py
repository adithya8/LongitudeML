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
import torch
import torch.nn as nn

from src import (
    get_default_args, get_logger,
    get_datasetDict, create_mask, MIDataLoaderModule,
    MILightningModule, recurrent, AutoRegressiveTransformer, PositionalEncoding
)

from CustomTransformers import TRNS_ARCHS      

if __name__ == '__main__':
    
    args = get_default_args()
    data = load_from_disk(args.data_dir)
    
    # data = data.rename_columns({'embeddings_lang': 'embeddings', 'mask_lang': 'mask'})
    # data = data.rename_columns({'embeddings_subscales': 'embeddings', 'mask_subscales': 'mask'})
    # data = data.rename_columns({'embeddings_hypLex': 'embeddings', 'mask_hypLex': 'mask'})
    
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

    if args.model_type == 'gru':
        model = recurrent(input_size = args.input_size, hidden_size = args.hidden_size, num_classes = args.num_classes, 
                                   num_outcomes = args.num_outcomes, num_layers = args.num_layers,  
                                   dropout = args.dropout, output_dropout=args.output_dropout, 
                                   bidirectional = args.bidirectional 
                                   )
    elif args.model_type == 'custom':
        if args.custom_model in ['langsubscaledualcontextformer']:
            lang_model = AutoRegressiveTransformer(input_size=args.input_size, hidden_size=args.hidden_size, num_classes=args.num_classes,
                                num_outcomes=args.num_outcomes, num_layers=args.num_layers,
                                dropout=args.dropout, output_dropout=args.output_dropout,
                                bidirectional=args.bidirectional, num_heads=args.num_heads, max_len=args.max_len
                                )
            subscales_model = AutoRegressiveTransformer(input_size=4, hidden_size=4, num_classes=args.num_classes,
                                num_outcomes=args.num_outcomes, num_layers=args.num_layers,
                                dropout=args.dropout, output_dropout=args.output_dropout,
                                bidirectional=args.bidirectional, num_heads=args.num_heads, max_len=args.max_len
                                )
            model = TRNS_ARCHS[args.custom_model](lang_model, subscales_model)
        elif args.custom_model in ['dailylangformer', 'pclsubscaleformer', 'totalpclformer', 'lextransformer', 'wtcpclsubscaleformer']:
            pcl_model = AutoRegressiveTransformer(input_size=args.input_size, hidden_size=args.hidden_size, num_classes=args.num_classes,
                                num_outcomes=args.num_outcomes, num_layers=args.num_layers,
                                dropout=args.dropout, output_dropout=args.output_dropout,
                                bidirectional=args.bidirectional, num_heads=args.num_heads, max_len=args.max_len
                                )
            model = TRNS_ARCHS[args.custom_model](pcl_model)
        else:
            raise ValueError('Invalid custom model: {}'.format(args.custom_model))        
        
    callbacks=[pl.callbacks.EarlyStopping(monitor="val_loss", patience=args.early_stopping_patience, 
                                        mode=args.early_stopping_mode, min_delta=args.early_stopping_min_delta)] if args.early_stopping_patience>0 else []
    callbacks.append(pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, save_last=False))

    trainer = pl.Trainer(accelerator='gpu', devices=1, default_root_dir=args.output_dir, logger=logger,
                        callbacks=callbacks, min_epochs=args.min_epochs, max_epochs=args.epochs)
        
    lightning_module = MILightningModule(args, model) 
    
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
    
"""
Example:

CUDA_VISIBLE_DEVICES=3 python run_PCL_forecast.py --data_dir /cronus_data/avirinchipur/ptsd_stop/forecasting/datasets/roberta_base_L11_rpca64_PCL_1_days_ahead_max90days_v3_40combined_5fold \
                        --output_dir /cronus_data/avirinchipur/ptsd_stop/forecasting/runs_1_ahead_only \
                        --overwrite_output_dir --num_outcomes 3 --model_type trns --num_layers 3 \
                        --num_heads 8 --max_len 90 --dropout 0.1 --output_dropout 0.1  --input_size 64 --lr 3e-4 \
                        --weight_decay 1e-2 --do_train --val_folds 0 --min_epochs 15 --epochs 40 \
                        --train_batch_size 16 --val_batch_size 32 --workspace ptsd-stop-forecasting --project_name 1_day_ahead \
                        --experiment_name dummy_trns_3lyr8hds_lr3e-4_wd1e-2_pe_1day_rpca64robaL11_max90_do0.1_opdo_0.1
"""