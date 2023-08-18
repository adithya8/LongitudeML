from typing import Any#, Optional
import torch
import torch.nn as nn
from pytorch_lightning.utilities.types import STEP_OUTPUT
# from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl

from .mi_model import recurrent

class MILightningModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        
        # TODO: Separate model_args from trainer_args
        self.args = args
        self.model = recurrent(input_size = args.input_size, hidden_size = args.hidden_size, num_classes = args.num_classes, \
                                 num_layers = args.num_layers, dropout = args.dropout, bidirectional = args.bidirectional)
        
        if args.num_classes>2:
            self.loss = nn.CrossEntropyLoss(weight=args.cross_entropy_class_weight)
        elif args.num_classes==2:
            self.loss = nn.BCEWithLogitsLoss()
        elif args.num_classes==1:
            self.loss = nn.MSELoss()
        else:
            raise ValueError("Invalid number of classes")
    
    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
    
    def unpack_batch_model_inputs(self, batch):
        batch_labels = batch.pop('labels', None)
        ids = batch.pop('id', None)
        seq_num = batch.pop('seq_num', None)
        predict_last_valid_hidden_state = batch.pop('predict_last_valid_hidden_state', False)
        batch.update({'predict_last_valid_hidden_state': predict_last_valid_hidden_state})
        return batch, batch_labels, ids, seq_num
    
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        batch, batch_labels, ids, seq_num = self.unpack_batch_model_inputs(batch)
        batch_output = self.model(**batch)
        batch_loss = self.loss(batch_output, batch_labels)
        self.log('train_loss', batch_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': batch_loss, 'pred': batch_output, 'labels': batch_labels, 'id': ids}
    
    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        batch, batch_labels, ids, seq_num = self.unpack_batch_model_inputs(batch)
        batch_output = self.model(**batch)
        batch_loss = self.loss(batch_output, batch_labels)
        self.log('val_loss', batch_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': batch_loss, 'pred': batch_output, 'labels': batch_labels, 'id': ids}
    
    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        batch, batch_labels, ids, seq_num = self.unpack_batch_model_inputs(batch)
        batch_output = self.model(**batch)
        batch_loss = self.loss(batch_output, batch_labels)
        self.log('test_loss', batch_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': batch_loss, 'pred': batch_output, 'labels': batch_labels, 'id': ids}

    def predict_step(self, batch, batch_idx) -> STEP_OUTPUT:
        batch, batch_labels, ids, seq_num = self.unpack_batch_model_inputs(batch)
        batch_output = self.model(**batch)
        # batch_loss = self.loss(batch_output, batch_labels)
        # self.log('val_loss', batch_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'pred': batch_output, 'labels': batch_labels, 'id': ids}
        
    #TODO: Log the loss values for all batches for viz
    def training_epoch_end(self, outputs) -> None:
        pass
    
    def validation_epoch_end(self, outputs) -> None:
        pass
    
    def test_epoch_end(self, outputs) -> None:
        pass
    