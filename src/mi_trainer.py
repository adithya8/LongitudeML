from typing import Any, Optional
import torch
import torch.nn as nn
from pytorch_lightning.utilities.types import STEP_OUTPUT
# from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl

from .mi_model import recurrent


class MITrainer(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        
        # TODO: Separate model_args from trainer_args
        self.args = args
        self.model = recurrent(input_size = args.input_size, hidden_size = args.hidden_size, num_classes = args.num_classes, \
                                 num_layers = args.num_layers, dropout = args.dropout, bidirectional = args.bidirectional)
        
        if args.num_classes>2:
            self.loss = nn.CrossEntropyLoss(weight = args.loss_weights)
        elif args.num_classes==2:
            self.loss = nn.BCEWithLogitsLoss()
        elif args.num_classes==1:
            self.loss = nn.MSELoss()
        else:
            raise ValueError("Invalid number of classes")
    
    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        batch_input, batch_target = batch
        batch_output = self.model(batch_input)
        batch_loss = self.loss(batch_output, batch_target)
        self.log('train_loss', batch_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': batch_loss, 'pred': batch_output, 'target': batch_target}
    
    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        batch_input, batch_target = batch
        batch_output = self.model(batch_input)
        batch_loss = self.loss(batch_output, batch_target)
        self.log('val_loss', batch_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': batch_loss, 'pred': batch_output, 'target': batch_target}
    
    def test_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT | None:
        return super().test_step(*args, **kwargs)

    def predict_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        return super().predict_step(*args, **kwargs)
        
            