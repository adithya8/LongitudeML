"""
    Utility functions for MI dataloading.
    Includes dataset creation, collation functions, Dataloader class that would be inputted to pl Trainer.
"""
from typing import Any, Dict, List, Union
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import DatasetDict, Dataset
import pytorch_lightning as pl


def get_dataset(data:Dict):
    """
        Returns the Huggingface datasets.arrow_dataset.Dataset
    """

    dataset = Dataset.from_dict(data)

    def create_defaut_time_ids(instance):
        """
            Creates a default time_ids for the instance assuming no breaks in timestep
        """
        instance['time_ids'] = list(range(len(instance['embeddings'][0])))
        return instance
    
    if 'time_ids' not in dataset.features:
        # TODO: Use Logger to log that default time ids are being created
        dataset = dataset.map(create_defaut_time_ids)
    
    return dataset


def get_datasetDict(train_data:Union[Dict, Dataset], val_data:Union[Dict, Dataset]=None, test_data:Union[Dict, Dataset]=None, val_folds:List=None, test_folds:List=None, fold_col:str='folds'):
    """
        Returns the Huggingface datasets.DatasetDict.
        Each input dictionary contains three key value pairs:
            1. embeddings: List of embeddings for each sequence of shape (1, seq_len, hidden_dim)
            2. labels: List of labels for each sequence of shape (seq_len, )
            3. time_ids [Optional]: List of sequence numbers for each sequence of shape (seq_len, )
    """
    
    datasetDict = DatasetDict()
    if train_data is not None: datasetDict['train'] = Dataset.from_dict(train_data) if isinstance(train_data, dict) else train_data
    if val_data is not None: datasetDict['val'] = Dataset.from_dict(val_data) if isinstance(val_data, dict) else val_data
    if test_data is not None: datasetDict['test']  = Dataset.from_dict(test_data) if isinstance(test_data, dict) else test_data

    if val_folds is not None:
        val_folds = set(val_folds)
        datasetDict['val'] = datasetDict['train'].filter(lambda example: example[fold_col] in val_folds).remove_columns(fold_col)
        datasetDict['train'] = datasetDict['train'].filter(lambda example: example[fold_col] not in val_folds).remove_columns(fold_col)
    
    if test_folds is not None:
        test_folds = set(test_folds)
        datasetDict['test'] = datasetDict['train'].filter(lambda example: example[fold_col] in test_folds).remove_columns(fold_col)
        datasetDict['train'] = datasetDict['train'].filter(lambda example: example[fold_col] not in test_folds).remove_columns(fold_col)
    
    def create_defaut_time_ids(instance):
        """
            Creates a default time_ids for the instance assuming no breaks in timestep
        """
        instance['time_ids'] = list(range(len(instance['embeddings'])))
        return instance
    
    for dataset_name in datasetDict:
        if 'time_ids' not in datasetDict[dataset_name].features:
            datasetDict[dataset_name] = datasetDict[dataset_name].map(create_defaut_time_ids)
        
    return datasetDict


def create_mask(examples):
    """
        Function that goes into DatasetDict.map() to create a mask pattern for the sequence
        This function is for applying MIL for a single label representing the sequence
    """
    def infill_missing_vector(instance):
        """
            Infills missing vector with the previous vector
            TODO: Other choices include default vector, neighbour average, past moving average, learnable embedding vector
        """
        sorted_time_ids = sorted(instance['time_ids'])
        missing_time_ids = set(range(sorted_time_ids[0], sorted_time_ids[-1]+1)) - set(sorted_time_ids)
        min_time_id = sorted_time_ids[0]
        original_time_id_mask = [1]*len(instance['time_ids'])
        infill_mask = [0]*len(instance['time_ids'])
        for time_id in sorted(missing_time_ids):
            # TODO: Following mask logic has to be changed in case we are going for last valid timestep prediction. 
            # The current logic in modelling uses the number of 1's in the mask to determine the last valid timestep.
            original_time_id_mask.insert(time_id-min_time_id, 1) # inserting 1 as mask at the missing time_id for EMI
            infill_mask.insert(time_id-min_time_id, 1) # inserting 1 as mask at the missing time_id 
            instance['time_ids'].insert(time_id-min_time_id, time_id)
            instance['query_ids'].insert(time_id-min_time_id, instance['query_ids'][time_id-min_time_id-1])
            # instance['embeddings'][0].insert(time_id-min_time_id, [0]*len(instance['embeddings'][0][time_id-min_time_id-1])) # 0 infilling is not better than copying the previous vector
            instance['embeddings'][0].insert(time_id-min_time_id, instance['embeddings'][0][time_id-min_time_id-1])
            instance['labels'].insert(time_id-min_time_id, instance['labels'][time_id-min_time_id-1])
        # instance['labels'] = torch.tensor(instance['labels']).expand(len(instance['time_idx'])).tolist()
        instance['mask'] = original_time_id_mask
        instance['infill_mask'] = infill_mask
        return instance

    def create_mask_pattern(instance):
        """
            Creates a mask pattern for the sequence
        """
        if 'mask' not in instance: 
            instance['mask'] = [1]*len(instance['time_ids'])# [1 if time_idx in instance['time_idx'] else 0 for time_idx in range(max(instance['time_idx'])+1)]
        return instance
    
    examples = infill_missing_vector(examples)
    examples = create_mask_pattern(examples)
    
    return examples


def default_collate_fn(features, predict_last_valid_timestep):
    # Features dict have embeddings, labels, time_ids, query_ids of single sequence (referenced by seq_idx)
    # predict_last_valid_timestep: True/False
    # Embeddings shape: (1, seq_len, hidden_dim)
    # Labels shape: (seq_len, )
    # time_idx shape: (seq_len, )
    # mask shape: (seq_len, )
    # infill_mask shape: (seq_len, )
    # query_id shape: (seq_len, )
    # seq_id shape: []
    first = features[0] # first is the first instance of the batch. features is a list of dictionary containing the instances. 
    if 'pad_mask' not in first: 
        for feat in features:
            feat['pad_mask'] = [0]*len(feat['time_ids'])
        first = features[0]
    
    batch = {"predict_last_valid_hidden_state": predict_last_valid_timestep }
    max_seq_len = max([len(feat['time_ids']) for feat in features])#+1
    seq_lens = [len(feat['time_ids']) for feat in features]
    num_outcomes = len(first['outcomes_mask'][0]) if 'outcomes_mask' in first else len(first['labels']) if 'labels' in first else len(first['outcomes'])
             
    for k, _ in first.items():
        if k.startswith("embeddings"):
            for feat in features:
                embeddings = torch.tensor(feat[k]).clone().detach()
                if len(embeddings.shape) == 2:
                    embeddings = embeddings.unsqueeze(0)
                elif len(embeddings.shape) != 3:
                    raise ValueError("Embeddings shape not supported. Expected shape (seq_len, hidden_dim) or (1, seq_len, hidden_dim). Got shape {}".format(embeddings.shape))
                if embeddings.shape[1] < max_seq_len:
                    zeros = torch.zeros((1, max_seq_len - embeddings.shape[1], embeddings.shape[2]))
                    embeddings = torch.cat((embeddings, zeros), dim=1)
                feat[k] = embeddings
            batch[k] = torch.cat([torch.tensor(f[k]) for f in features], dim=0)
        elif k == "labels" or k == "outcomes" or k == "outcomes_mask":
            for feat in features:
                outcomes = torch.tensor(feat[k]).clone().detach() 
                if len(outcomes.shape) == 2:
                    outcomes = outcomes.unsqueeze(0)
                elif len(outcomes.shape) != 3:
                    raise ValueError("Outcomes shape not supported. Expected shape (seq_len, outcomes_dim) or (1, seq_len, outcomes_dim). Got shape {}".format(outcomes.shape))
                if outcomes.shape[1] < max_seq_len:
                    zeros = torch.zeros((1, max_seq_len - outcomes.shape[1], outcomes.shape[2]))
                    outcomes = torch.cat((outcomes, zeros), dim=1)
                feat[k] = outcomes
            batch[k] = torch.cat([torch.tensor(f[k]) for f in features], dim=0)
            if k=="outcomes_mask": batch[k] = batch[k].to(torch.bool) 
        elif k=="pad_mask" or k=="time_ids" or k=="infill_mask" or k.startswith("mask"):
            padding_value = -1 if k=="time_ids" else (1 if k=="pad_mask" or k.startswith("mask") else 0)
            batch[k] = torch.nn.utils.rnn.pad_sequence([torch.tensor(f[k]) for f in features], padding_value=padding_value, batch_first=True)
            if k == "pad_mask" or k == "infill_mask" or k.startswith("mask") or k=="outcomes_mask": 
                batch[k] = batch[k].to(torch.bool)
            elif k == "time_ids":
                batch[k] = batch[k].to(torch.long)
        elif k=="query_ids":
            batch[k] = torch.nn.utils.rnn.pad_sequence([torch.tensor(f[k]) for f in features], padding_value=-1, batch_first=True)
        elif k=="seq_idx" or k=="seq_id":
            batch[k] = torch.tensor([f[k] for f in features]).reshape(len(features), -1)
        else:
            pass
            # raise Warning("Key {} not supported for batching. Leaving it out of the dataloaders".format(k))        
    
    return batch


class MIDataLoaderModule(pl.LightningDataModule):
    """
        Data loader module for MI. Takes the MI dataset as input.
    """
    def __init__(self, data_args, datasets: Dict[str, Any]):
        super().__init__()
        self.args = data_args
        self.train_dataset = datasets.pop('train', None)
        self.val_dataset = datasets.pop('val', None)
        self.test_dataset = datasets.pop('test', None)
        self.predict_dataset = datasets.pop('predict', None)
        print ('Predict Last Valid Timestep set to {}'.format(data_args.predict_last_valid_timestep))
        self.collate_fn = lambda b: default_collate_fn(b, data_args.predict_last_valid_timestep) # NOTE: Either change to class method or use partial in case more args are needed

    def train_dataloader(self):
        if self.train_dataset is None: return None
        return DataLoader(self.train_dataset, batch_size=self.args.train_batch_size, shuffle=True, collate_fn=self.collate_fn)#, num_workers=self.args.num_workers)

    def val_dataloader(self):
        if self.val_dataset is None: return None
        return DataLoader(self.val_dataset, batch_size=self.args.val_batch_size, shuffle=False, collate_fn=self.collate_fn)#, num_workers=self.args.num_workers)

    def test_dataloader(self):
        if self.test_dataset is None: return None
        return DataLoader(self.test_dataset, batch_size=self.args.val_batch_size, shuffle=False, collate_fn=self.collate_fn)#, num_workers=self.args.num_workers)

    def predict_dataloader(self):
        if self.predict_dataset is None: return None
        return DataLoader(self.predict_dataset, batch_size=self.args.val_batch_size, shuffle=False, collate_fn=self.collate_fn)#, num_workers=self.args.num_workers,