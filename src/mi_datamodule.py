"""
    Utility functions for MI dataloading.
    Includes dataset creation, collation functions, Dataloader class that would be inputted to pl Trainer.
"""
from typing import Any, Dict
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import DatasetDict, Dataset
import pytorch_lightning as pl


def get_datasetDict(train_data:Dict, val_data:Dict, test_data:Dict):
    """
        Returns the Huggingface datasets.DatasetDict.
        Each input dictionary contains three key value pairs:
            1. embeddings: List of embeddings for each sequence of shape (1, seq_len, hidden_dim)
            2. labels: List of labels for each sequence of shape (seq_len, )
            3. time_idx [Optional]: List of sequence numbers for each sequence of shape (seq_len, )
    """
    
    datasetDict = DatasetDict()
    if train_data is not None: datasetDict['train'] = Dataset.from_dict(train_data)
    if val_data is not None: datasetDict['dev'] = Dataset.from_dict(val_data)
    if test_data is not None: datasetDict['test']  = Dataset.from_dict(test_data)
    
    def create_defaut_time_ids(instance):
        """
            Creates a default time_ids for the instance assuming no breaks in timestep
        """
        instance['time_ids'] = list(range(len(instance['embeddings'][0])))
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
        for time_id in sorted(missing_time_ids):
            # TODO: Following mask logic has to be changed in case we are going for last valid timestep prediction. 
            # The current logic in modelling uses the number of 1's in the mask to determine the last valid timestep.
            original_time_id_mask.insert(time_id-min_time_id, 0) # inserting 0 at the missing time_id
            instance['time_ids'].insert(time_id-min_time_id, time_id)
            instance['query_ids'].insert(time_id-min_time_id, instance['query_ids'][time_id-min_time_id-1])
            instance['embeddings'][0].insert(time_id-min_time_id, instance['embeddings'][0][time_id-min_time_id-1])
            instance['labels'].insert(time_id-min_time_id, instance['labels'][time_id-min_time_id-1])
        # instance['labels'] = torch.tensor(instance['labels']).expand(len(instance['time_idx'])).tolist()
        instance['mask'] = original_time_id_mask
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
    # query_id shape: (seq_len, )
    # seq_id shape: []
    first = features[0]
    batch = {"predict_last_valid_hidden_state": predict_last_valid_timestep }
    max_seq_len = max([len(feat['time_ids']) for feat in features])#+1
    for k, _ in first.items():
        if k == "embeddings":
            for feat in features:
                embeddings = torch.tensor(feat['embeddings']).clone()
                if embeddings.shape[1] < max_seq_len:
                    zeros = torch.zeros((1, max_seq_len - embeddings.shape[1], embeddings.shape[2]))
                    feat['embeddings'] = torch.cat((embeddings, zeros), dim=1)
            batch[k] = torch.cat([torch.tensor(f[k]) for f in features], dim=0)
        elif k=="labels" or k=="mask" or k=="time_ids":
            batch[k] = torch.nn.utils.rnn.pad_sequence([torch.tensor(f[k]) for f in features], padding_value=0, batch_first=True)
            if k == "mask": 
                batch[k] = batch[k].to(torch.bool)
            elif k == "labels": 
                batch[k] = batch[k].to(torch.float)
            elif k == "time_ids":
                batch[k] = batch[k].to(torch.long)
        elif k=="query_ids":
            batch[k] = torch.nn.utils.rnn.pad_sequence([torch.tensor(f[k]) for f in features], padding_value=-1, batch_first=True)
        elif k=="seq_idx":
            batch[k] = torch.tensor([f[k] for f in features])
        else:
            raise Warning("Key {} not supported for batching. Leaving it out of the dataloaders".format(k))
    return batch


class MIDataLoaderModule(pl.LightningDataModule):
    """
        Data loader module for MI. Takes the MI dataset as input.
    """
    def __init__(self, data_args, datasets: Dict[str, Any]):
        super().__init__()
        self.args = data_args
        self.train_dataset = datasets.pop('train', None)
        self.dev_dataset = datasets.pop('dev', None)
        self.test_dataset = datasets.pop('test', None)
        self.predict_dataset = datasets.pop('predict', None)
        print ('Predict Last Valid Timestep set to {}'.format(data_args.predict_last_valid_timestep))
        self.collate_fn = lambda b: default_collate_fn(b, data_args.predict_last_valid_timestep) # NOTE: Either change to class method or use partial in case more args are needed        
        
    # def prepare_data(self):
    #     """
    #         This method is used to download and prepare the data.
    #     """
    #     pass

    # def setup(self):
    #     """
    #         This method is used to load the data.
    #     """
    #     pass

    def train_dataloader(self):
        if self.train_dataset is None: return None
        return DataLoader(self.train_dataset, batch_size=self.args.train_batch_size, shuffle=False, collate_fn=self.collate_fn)#, num_workers=self.args.num_workers)

    def val_dataloader(self):
        if self.dev_dataset is None: return None
        return DataLoader(self.dev_dataset, batch_size=self.args.val_batch_size, shuffle=False, collate_fn=self.collate_fn)#, num_workers=self.args.num_workers)

    def test_dataloader(self):
        if self.test_dataset is None: return None
        return DataLoader(self.test_dataset, batch_size=self.args.val_batch_size, shuffle=False, collate_fn=self.collate_fn)#, num_workers=self.args.num_workers)

    def predict_dataloader(self):
        if self.predict_dataset is None: return None
        return DataLoader(self.predict_dataset, batch_size=self.args.val_batch_size, shuffle=False, collate_fn=self.collate_fn)#, num_workers=self.args.num_workers,