"""
    Utility functions for MI dataloading.
    Includes dataset creation, collation functions, Dataloader class that would be inputted to pl Trainer.
"""
from typing import Any, Dict
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import DatasetDict, Dataset
import pytorch_lightning as pl


def get_dataset(train_data:dict, val_data:dict, test_data:dict):
    """
        Returns the Huggingface datasets.DatasetDict .
        Each input dictionary contains three key value pairs:
            1. embeddings: List of embeddings for each sequence of shape (1, seq_len, hidden_dim)
            2. labels: List of labels for each sequence of shape (1, )
            3. seq_num [Optional]: List of sequence numbers for each sequence of shape (seq_len, )
    """
    
    datasetDict = DatasetDict()
    datasetDict['train'] = Dataset.from_dict(train_data)
    datasetDict['dev'] = Dataset.from_dict(val_data)
    datasetDict['test']  = Dataset.from_dict(test_data)
    
    def create_defaut_seq_num(instance):
        """
            Creates a default seq_num for the instance assuming no breaks in timestep
        """
        instance['seq_num'] = list(range(len(instance['embeddings'][0])))
        return instance
    
    for dataset_name in datasetDict:
        if 'seq_num' not in datasetDict[dataset_name].features:
            datasetDict[dataset_name] = datasetDict[dataset_name].map(create_defaut_seq_num)
        
    return datasetDict


def create_mask(examples):
    """
        Function that goes into DatasetDict.map() to create a mask pattern for the sequence
        This function is for applying MIL for a single label representing the sequence
    """
    def infill_missing_vector(instance):
        """
            Infills missing vector with the previous vector
            TODO: Other choices include default vector, neighbour average, learnable embedding vector
        """
        sorted_seq_nums = sorted(instance['seq_num'])
        missing_seq_nums = set(range(sorted_seq_nums[0], sorted_seq_nums[-1]+1)) - set(sorted_seq_nums)
        for seq_num in missing_seq_nums:
            instance['embeddings'][0].insert(seq_num, instance['embeddings'][0][seq_num-1])
            instance['seq_num'].insert(seq_num, seq_num)
        instance['labels'] = torch.tensor(instance['labels']).expand(len(instance['seq_num'])).tolist()
        return instance

    def create_mask_pattern(instance):
        """
            Creates a mask pattern for the sequence
        """
        instance['mask'] = [1]*len(instance['seq_num'])# [1 if seq_num in instance['seq_num'] else 0 for seq_num in range(max(instance['seq_num'])+1)]
        return instance
    
    examples = infill_missing_vector(examples)
    examples = create_mask_pattern(examples)
    
    return examples  


def default_collate_fn(features):
    # Features dict have embeddings, label, seq_num of single sequence
    # Embeddings shape: (1, seq_len, hidden_dim)
    # Labels shape: (seq_len, )
    # seq_num shape: (seq_len, )
    # mask shape: (seq_len, )
    first = features[0]
    batch = {}
    max_seq_len = max([feat['seq_num'][-1] for feat in features])+1
    for k, _ in first.items():
        if k == "embeddings":
            for feat in features:
                embeddings = torch.tensor(feat['embeddings']).clone()
                if embeddings.shape[1] < max_seq_len:
                    zeros = torch.zeros((1, max_seq_len - embeddings.shape[1], embeddings.shape[2]))
                    feat['embeddings'] = torch.cat((embeddings, zeros), dim=1)
            batch[k] = torch.cat([torch.tensor(f[k]) for f in features], dim=0)
        elif k=="labels" or k=="mask" or k=="seq_num":
            batch[k] = torch.nn.utils.rnn.pad_sequence([torch.tensor(f[k]) for f in features], padding_value=0, batch_first=True)
        else:
            raise NotImplementedError(f"Key {k} not supported for batching.")
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
        self.collate_fn = default_collate_fn            
        
    def prepare_data(self):
        """
            This method is used to download and prepare the data.
        """
        # raise NotImplementedError("You need to implement the prepare_data() method")
        pass

    def setup(self):
        """
            This method is used to load the data.
        """
        pass

    def train_dataloader(self):
        if self.train_dataset is None: return None
        return DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, collate_fn=self.collate_fn)

    def val_dataloader(self):
        if self.dev_dataset is None: return None
        return DataLoader(self.dev_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, collate_fn=self.collate_fn)

    def test_dataloader(self):
        if self.test_dataset is None: return None
        return DataLoader(self.test_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, collate_fn=self.collate_fn)

    def predict_dataloader(self):
        if self.predict_dataset is None: return None
        return DataLoader(self.predict_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, collate_fn=self.collate_fn)