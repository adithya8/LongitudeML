from typing import Any, Dict
# import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

from .mi_datacollator import MIDataCollator


class MIDatasetModule(Dataset):
    """
        Dataset module for MI. 
    """
    def __init__(self, embeddings=None, cls=None, labels=None, model_input_type="embeddings+cls") -> None:
        super().__init__()
        self.embeddings = embeddings
        self.cls = cls 
        self.labels = labels
        self.model_input_type = model_input_type
        self.len = len(self.embeddings) if "embeddings" in self.model_input_type else len(self.cls)
        
    def __len__(self) -> int:
        return self.len
    
    def __getitem__(self, index: int) -> Any:
        if self.model_input_type == "embeddings+cls":
            return self.embeddings[index], self.cls[index], self.labels[index]
        elif self.model_input_type == "embeddings":
            return self.embeddings[index], self.labels[index]
        elif self.model_input_type == "cls":
            return self.cls[index], self.labels[index]
        
        
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
        self.collate_fn = None if self.args.collate_fn is None else MIDataCollator(getattr(self, self.args.collate_fn))
        
    def prepare_data(self):
        """
            This method is used to download and prepare the data.
        """
        # raise NotImplementedError("You need to implement the prepare_data() method")
        pass

    def setup(self, stage=None):
        """
            This method is used to load the data.
        """
        pass

    def train_dataloader(self):
        #TODO: Call datacollator here
        if self.train_dataset is None: return None
        return DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)

    def val_dataloader(self):
        if self.dev_dataset is None: return None
        return DataLoader(self.dev_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)

    def test_dataloader(self):
        if self.test_dataset is None: return None
        return DataLoader(self.test_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)

    def predict_dataloader(self):
        if self.predict_dataset is None: return None
        return DataLoader(self.predict_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)