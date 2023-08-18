from typing import Any, Optional, Dict
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT

from .mi_model import recurrent


class MILightningModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        
        # TODO: Separate model_args from trainer_args (May be)
        self.args = args
        # TODO: Move model instantiation outside LightningModule. 
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

        # Store step outputs
        self.step_outputs = dict(train=[], val=[], test=[])
        # Store avg loss over epochs
        self.epoch_loss = dict(train=[], val=[], test=[])
    
    
    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
    
    
    def unpack_batch_model_inputs(self, batch):
        """
            Unpack the batch and return the model inputs separated from the other batch tensors
        """
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
        # self.log('train_loss', batch_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        step_output = {'loss': batch_loss, 'pred': batch_output, 'labels': batch_labels}
        if ids is not None: step_output.update({'id': ids})
        self.step_outputs['train'].append(step_output) 
        return step_output
    
    
    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        batch, batch_labels, ids, seq_num = self.unpack_batch_model_inputs(batch)
        batch_output = self.model(**batch)
        batch_loss = self.loss(batch_output, batch_labels)
        # self.log('val_loss', batch_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        step_output = {'loss': batch_loss, 'pred': batch_output, 'labels': batch_labels}
        if ids is not None: step_output.update({'id': ids})
        self.step_outputs['val'].append(step_output) 
        return step_output
    
    
    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        batch, batch_labels, ids, seq_num = self.unpack_batch_model_inputs(batch)
        batch_output = self.model(**batch)
        batch_loss = self.loss(batch_output, batch_labels)
        # self.log('test_loss', batch_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        step_output = {'loss': batch_loss, 'pred': batch_output, 'labels': batch_labels}
        if ids is not None: step_output.update({'id': ids})
        self.step_outputs['test'].append(step_output) 
        return step_output


    def predict_step(self, batch, batch_idx) -> STEP_OUTPUT:
        batch, batch_labels, ids, seq_num = self.unpack_batch_model_inputs(batch)
        batch_output = self.model(**batch)
        return {'pred': batch_output, 'labels': batch_labels, 'id': ids}


    def process_epoch_end(self, step_outputs) -> Dict:
        
        first = step_outputs[0]
        keys = first.keys()
        cat_outputs = dict([(k,[]) for k in keys])
        for _, step_output in enumerate(step_outputs):
            for k in keys:
                cat_outputs[k].append(step_output[k].view(-1, 1))         

        for k in keys:
            cat_outputs[k] = torch.cat(cat_outputs[k], dim=0)
            
        return cat_outputs 
             
             
    #TODO: Log the loss values for all batches for viz
    def on_train_epoch_end(self) -> None:
        if self.global_rank==0:
            cat_outputs = self.process_epoch_end(self.step_outputs['train'])
            if 'loss' in cat_outputs:
                avg_loss = torch.mean(cat_outputs['loss']) 
                self.log('train_epoch_loss', avg_loss, on_epoch=True, prog_bar=True, logger=True)
                self.epoch_loss['train'].append(avg_loss.item())
            self.step_outputs['train'].clear()

    
    def on_validation_epoch_end(self) -> None:
        if self.global_rank==0:
            cat_outputs = self.process_epoch_end(self.step_outputs['val'])
            if 'loss' in cat_outputs:
                avg_loss = torch.mean(cat_outputs['loss']) 
                self.log('val_epoch_loss', avg_loss, on_epoch=True, prog_bar=True, logger=True)
                self.epoch_loss['val'].append(avg_loss.item())
            self.step_outputs['val'].clear()
            
    
    def on_test_epoch_end(self, outputs) -> None:
        if self.global_rank==0:
            cat_outputs = self.process_epoch_end(self.step_outputs['test'])
            if 'loss' in cat_outputs:
                avg_loss = torch.mean(cat_outputs['loss']) 
                self.log('test_epoch_loss', avg_loss, on_epoch=True, prog_bar=True, logger=True)
                self.epoch_loss['test'].append(avg_loss.item())
            self.step_outputs['test'].clear()
    
    
    # def predict_epoch_end(self, outputs) -> None:
    #     if self.global_rank==0:
    #         cat_outputs = self.process_epoch_end(outputs)
    