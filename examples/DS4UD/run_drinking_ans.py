import os, pickle
from utils import add_to_path
add_to_path(__file__)

import pytorch_lightning as pl

from src import (
    get_default_args, get_logger,
    get_datasetDict, create_mask, MIDataLoaderModule,
    MILightningModule
)


if __name__ == '__main__':

    # Get the args from command line
    args = get_default_args()

    # Get the logger and log the hyperparameters
    logger = get_logger('comet', workspace=args.workspace, project_name=args.project_name, experiment_name=args.experiment_name, save_dir=args.output_dir)
    logger.log_hyperparams(args.__dict__)

    # Set the seed for reproducibility
    pl.seed_everything(args.seed)

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
        if args.train_file is None and args.do_train:
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

    if ('train_data' not in dataDict or not dataDict['train_data']) and args.do_train:
        print ('train_data not provided. Training will not be performed.')
        dataDict['train_data'] = None
    if ('val_data' not in dataDict or not dataDict['val_data']):
        print ('val_data not provided. Validation split will be performed.')
        dataDict['val_data'] = None
    if ('test_data' not in dataDict or not dataDict['test_data']):
        if args.do_test: print ('test_data not provided. Testing will not be performed.')
        dataDict['test_data'] = None
    
    # TODO: the function expects train, val and test. Correct this. 
    # Get the datasetDict (in HF datasetDict format) containing the embeddings across the temporal dimension for each sequence along with the labels and sequence numbers
    datasetDict = get_datasetDict(train_data=dataDict['train_data'], val_data=dataDict['val_data'], test_data=dataDict['test_data'])
    
    # Create a mask pattern for the sequence
    datasetDict = datasetDict.map(create_mask)
    
    # Get the dataloader module
    dataloaderModule = MIDataLoaderModule(args, datasetDict)
    
    if not os.path.exists(args.output_dir):
        print ('Creating output directory: {}'.format(args.output_dir))
        os.makedirs(args.output_dir)
    
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=args.epochs, default_root_dir=args.output_dir, logger=logger)
    
    lightning_module = MILightningModule(args) 
    
    # TODO: id throuwing error in batching
    # Error:   File "/users2/avirinchipur/NLP/emi/src/mi_datamodule.py", line 96, in default_collate_fn
    # raise Warning(f"Key {k} not supported for batching. Leaving it out of the dataloaders")
    # TODO: Also replace all "Warning" with Log
    trainer.fit(lightning_module, train_dataloaders=dataloaderModule.train_dataloader(), val_dataloaders=dataloaderModule.val_dataloader())
    
    logger.experiment.end()