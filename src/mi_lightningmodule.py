from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT

from .mi_model import recurrent
from .mi_eval import mi_mse, mi_smape

class MILightningModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        
        # TODO: Separate model_args from trainer_args (May be)
        self.args = args
        # TODO: Move model instantiation outside LightningModule. 
        self.model = recurrent(input_size = args.input_size, hidden_size = args.hidden_size, num_classes = args.num_classes, \
                                 num_layers = args.num_layers, dropout = args.dropout, bidirectional = args.bidirectional)
        
        self.metrics = {}
        if args.num_classes>2:
            self.loss = nn.CrossEntropyLoss(weight=args.cross_entropy_class_weight)
        elif args.num_classes==2:
            self.loss = nn.BCEWithLogitsLoss()
        elif args.num_classes==1:
            self.loss = mi_mse #nn.MSELoss()
            self.metrics = {'smape': mi_smape}
        else:
            raise ValueError("Invalid number of classes")

        # Store step outputs
        self.step_outputs = dict(train=[], val=[], test=[])
        # Store avg loss over epochs
        self.epoch_loss = dict(train=[], val=[], test=[])
        # Store predictions and targets. Keys: preds, target since pytorch metrics use these arguments
        self.labels = dict(train=dict(preds=[], target=[]), val=dict(preds=[], target=[]), test=dict(preds=[], target=[]))
        # self.labels = dict(preds=dict(train=[], val=[], test=[]), target=dict(train=[], val=[], test=[]))
    
    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
    
    
    def unpack_batch_model_inputs(self, batch):
        """
            Unpack the batch and return the model inputs separated from the other batch tensors
        """
        batch_labels = batch.pop('labels', None)
        query_id = batch.pop('query_ids', None)
        seq_id = batch.pop('seq_idx', None)
        time_idx = batch.pop('time_ids', None)
        return batch, batch_labels, seq_id, time_idx
    
    
    def calculate_metrics(self, input:torch.Tensor, target:torch.Tensor, mask:torch.Tensor=None) -> Dict:
        metric_score = dict()
        for metric in self.metrics:
            metric_score[metric] = self.metrics[metric](input, target, mask)
            
        return metric_score
    
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        batch, batch_labels, seq_id, time_idx = self.unpack_batch_model_inputs(batch)
        batch_output = self.model(**batch)
        # Loss only calculates for the valid timesteps 
        batch_loss = self.loss(input=batch_output, target=batch_labels, mask=batch['mask'])
        step_metrics = self.calculate_metrics(batch_output, batch_labels, batch['mask'])
        # TODO: Fix step level metric logging. Train logging is probably okay, val logging shows opposite trend b/w step and epoch 
        # self.log('train_loss', batch_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.logger.log_metrics({'train_loss': batch_loss}, step = self.global_step)
        step_output = {'loss': batch_loss, 'pred': batch_output, 'labels': batch_labels, 'step': torch.tensor([self.global_step])}
        if seq_id is not None: step_output.update({'seq_id': seq_id})
        step_output.update(step_metrics)
        self.step_outputs['train'].append(step_output) 
        return step_output
    
    
    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        batch, batch_labels, seq_id, time_idx = self.unpack_batch_model_inputs(batch)
        batch_output = self.model(**batch)
        batch_loss = self.loss(input=batch_output, target=batch_labels, mask=batch['mask'])
        step_metrics = self.calculate_metrics(batch_output, batch_labels, batch['mask'])
        # self.log('val_loss', batch_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.logger.log_metrics({'val_loss': batch_loss}, step=self.global_step)
        step_output = {'loss': batch_loss, 'pred': batch_output, 'labels': batch_labels, 'step': torch.tensor([self.global_step])}
        if seq_id is not None: step_output.update({'seq_id': seq_id})
        step_output.update(step_metrics)
        self.step_outputs['val'].append(step_output) 
        return step_output
    
    
    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        batch, batch_labels, seq_id, time_idx = self.unpack_batch_model_inputs(batch)
        batch_output = self.model(**batch)
        batch_loss = self.loss(input=batch_output, target=batch_labels)
        step_metrics = self.calculate_metrics(batch_output, batch_labels)
        # self.log('test_loss', batch_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        step_output = {'loss': batch_loss, 'pred': batch_output, 'labels': batch_labels}
        if seq_id is not None: step_output.update({'seq_id': seq_id})
        step_output.update(step_metrics)
        self.step_outputs['test'].append(step_output) 
        return step_output


    def predict_step(self, batch, batch_idx) -> STEP_OUTPUT:
        batch, batch_labels, seq_id, time_idx = self.unpack_batch_model_inputs(batch)
        batch_output = self.model(**batch)
        return {'pred': batch_output, 'labels': batch_labels, 'seq_id': seq_id}


    def process_epoch_end(self, step_outputs) -> Dict:
        """
            Process the step outputs and return the epoch outputs concatenated across all steps
            Stores loss, predictions and labels as (num_steps, 1)
        """
        first = step_outputs[0]
        keys = first.keys()
        cat_outputs = dict([(k,[]) for k in keys])
        for _, step_output in enumerate(step_outputs):
            for k in keys:
                cat_outputs[k].append(step_output[k].view(-1, 1))         

        for k in keys:
            cat_outputs[k] = torch.cat(cat_outputs[k], dim=0)
            
        return cat_outputs 
         

    def save_outputs(self, process:str) -> None:
        cat_outputs = self.process_epoch_end(self.step_outputs[process])
        if 'loss' in cat_outputs:
            avg_loss = torch.mean(cat_outputs['loss']) 
            self.epoch_loss[process].append(avg_loss.item())
            print ("{} loss epoch {}: {}".format(process, self.current_epoch, avg_loss))
        if 'pred' in cat_outputs:
            self.labels[process]['preds'].append(cat_outputs['pred'])
        if 'labels' in cat_outputs:
            self.labels[process]['target'].append(cat_outputs['labels'])
        # ADd metrics to this
            

    def on_train_epoch_end(self) -> None:
        if self.global_rank==0:
            self.save_outputs(process="train")
            self.logger.log_metrics({'train_epoch_loss': self.epoch_loss['train'][-1]}, step=self.current_epoch)
            # self.log('train_epoch_loss', self.epoch_loss['train'][-1], on_epoch=True, prog_bar=True, logger=True)
            self.step_outputs['train'].clear()

    
    def on_validation_epoch_end(self) -> None:
        if self.global_rank==0:
            self.save_outputs(process="val")
            self.logger.log_metrics({'val_epoch_loss': self.epoch_loss['val'][-1]}, step=self.current_epoch)
            # self.log('val_epoch_loss', self.epoch_loss['val'][-1], on_epoch=True, prog_bar=True, logger=True)
            self.step_outputs['val'].clear()
            
    
    def on_test_epoch_end(self) -> None:
        if self.global_rank==0:
            self.save_outputs(process="test")
            self.logger.log_metrics({'test_epoch_loss': self.epoch_loss['test'][-1]}, step=self.current_epoch)
            # self.log('test_epoch_loss', self.epoch_loss['test'][-1], on_epoch=True, prog_bar=True, logger=True)
            self.step_outputs['test'].clear()
    
    
    #TODO: Implement on_predict_epoch_end/predict_epoch_end
    