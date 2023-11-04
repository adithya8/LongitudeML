from typing import Any, Dict
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT

from .mi_model import recurrent
from .mi_eval import mi_mse, mi_smape, mi_pearsonr, mi_mae

class MILightningModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        
        # TODO: Separate model_args from trainer_args (May be)
        self.args = args
        # TODO: Move model instantiation outside LightningModule. 
        self.model = recurrent(input_size = args.input_size, hidden_size = args.hidden_size, num_classes = args.num_classes, \
                                 num_layers = args.num_layers, dropout = args.dropout, output_dropout=args.output_dropout, \
                                 bidirectional = args.bidirectional)
        
        self.metrics_fns = {}
        if args.num_classes>2:
            self.loss = nn.CrossEntropyLoss(weight=args.cross_entropy_class_weight)
        elif args.num_classes==2:
            self.loss = nn.BCEWithLogitsLoss()
        elif args.num_classes==1:
            self.loss = mi_mse #nn.MSELoss()
            self.metrics_fns = {'smape': mi_smape, 'pearsonr': mi_pearsonr, 'mae': mi_mae}
        else:
            raise ValueError("Invalid number of classes")

        # Store step outputs
        self.step_outputs = dict(train=[], val=[], test=[])
        # Store avg loss over epochs
        self.epoch_loss = dict(train=[], val=[], test=[])
        # Store the metrics for each epoch for each of train, dev and test
        process_metric_dict = dict([(i, []) for i in self.metrics_fns])
        self.epoch_metrics = dict(train=process_metric_dict.copy(), val=process_metric_dict.copy(), test=process_metric_dict.copy())
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
        for metric_name, fn in self.metrics_fns.items():
            metric_score[metric_name] = fn(input=input, target=target, mask=mask)
        return metric_score

    
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        batch, batch_labels, seq_id, time_idx = self.unpack_batch_model_inputs(batch)
        batch_output = self.model(**batch)
        # Loss only calculates for the valid timesteps 
        batch_loss = self.loss(input=batch_output, target=batch_labels, mask=batch['mask'])
        step_metrics = self.calculate_metrics(batch_output, batch_labels, batch['mask'])
        log_metrics_dict = {'train_loss': batch_loss}
        log_metrics_dict.update({'train_{}'.format(key): val for key, val in step_metrics.items()})
        self.log_dict(log_metrics_dict, on_step=False, on_epoch=True)
        # TODO: Fix step level metric logging. Train logging is probably okay, val logging shows opposite trend b/w step and epoch 
        # self.log('train_loss', batch_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.logger.log_metrics({'train_loss': batch_loss}, step = self.global_step)
        step_output = {'loss': batch_loss, 'pred': batch_output, 'labels': batch_labels, 'mask': batch['mask'], 'step': torch.tensor([self.global_step])}
        if seq_id is not None: step_output.update({'seq_id': seq_id})
        step_output.update(step_metrics)
        self.step_outputs['train'].append(step_output)
        return step_output
    
    
    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        batch, batch_labels, seq_id, time_idx = self.unpack_batch_model_inputs(batch)
        batch_output = self.model(**batch)
        batch_loss = self.loss(input=batch_output, target=batch_labels, mask=batch['mask'])
        step_metrics = self.calculate_metrics(batch_output, batch_labels, batch['mask'])
        log_metrics_dict = {'val_loss': batch_loss}
        log_metrics_dict.update({'val_{}'.format(key): val for key, val in step_metrics.items()})
        self.log_dict(log_metrics_dict, on_step=False, on_epoch=True)
        # self.logger.log_metrics({'val_loss': batch_loss}, step=self.global_step)
        step_output = {'loss': batch_loss, 'pred': batch_output, 'labels': batch_labels, 'mask': batch['mask'], 'step': torch.tensor([self.global_step])}
        if seq_id is not None: step_output.update({'seq_id': seq_id})
        step_output.update(step_metrics)
        self.step_outputs['val'].append(step_output) 
        return step_output
    
    
    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        batch, batch_labels, seq_id, time_idx = self.unpack_batch_model_inputs(batch)
        batch_output = self.model(**batch)
        batch_loss = self.loss(input=batch_output, target=batch_labels)
        step_metrics = self.calculate_metrics(batch_output, batch_labels)
        log_metrics_dict = {'test_loss': batch_loss}
        log_metrics_dict.update({'test_{}'.format(key): val for key, val in step_metrics.items()})
        self.log_dict(log_metrics_dict, on_step=False, on_epoch=True)
        # self.log('test_loss', batch_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        step_output = {'loss': batch_loss, 'pred': batch_output, 'labels': batch_labels, 'mask': batch['mask']}
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
                cat_outputs[k].append(step_output[k].detach().clone().cpu())
            
        return cat_outputs 
         

    def save_outputs(self, process:str) -> None:
        cat_outputs = self.process_epoch_end(self.step_outputs[process])
        if 'loss' in cat_outputs:
            avg_loss = torch.tensor(cat_outputs['loss']).mean()
            self.epoch_loss[process].append(avg_loss.item())
            # print ("{} loss epoch {}: {}".format(process, self.current_epoch, avg_loss))
        # Only store the predictions and labels for the valid timestep
        if 'pred' in cat_outputs:
            self.labels[process]['preds'].append(cat_outputs['pred'])
        if 'labels' in cat_outputs:
            self.labels[process]['target'].append(cat_outputs['labels'])
        if self.metrics_fns:
            pred_flatten = torch.cat([i.view(-1, 1) for i in cat_outputs['pred']], dim=0)
            target_flatten = torch.cat([i.view(-1, 1) for i in cat_outputs['labels']], dim=0)
            mask_flatten = torch.cat([i.view(-1, 1) for i in cat_outputs['mask']], dim=0)
            for metric_name, fn in self.metrics_fns.items():
                self.epoch_metrics[process][metric_name].append(fn(input=pred_flatten, target=target_flatten, mask=mask_flatten))


    def on_train_epoch_end(self) -> None:
        if self.global_rank==0:
            self.save_outputs(process="train")
            # TODO: We want epoch level flattened metric
            self.logger.log_metrics({'train_epoch_loss': self.epoch_loss['train'][-1]}, epoch=self.current_epoch)
            for metric_name in self.metrics_fns:
                self.logger.log_metrics({'train_epoch_{}'.format(metric_name): self.epoch_metrics['train'][metric_name][-1]}, epoch=self.current_epoch)
            # self.log('train_epoch_loss', self.epoch_loss['train'][-1], on_epoch=True, prog_bar=True, logger=True)
            self.step_outputs['train'].clear()

    
    def on_validation_epoch_end(self) -> None:
        if self.global_rank==0:
            self.save_outputs(process="val")
            self.logger.log_metrics({'val_epoch_loss': self.epoch_loss['val'][-1]}, epoch=self.current_epoch)
            for metric_name in self.metrics_fns:
                self.logger.log_metrics({'val_epoch_{}'.format(metric_name): self.epoch_metrics['val'][metric_name][-1]}, epoch=self.current_epoch)
            #   # self.log_dict({'val_epoch_{}'.format(metric_name): self.epoch_metrics['val'][metric_name][-1]}, on_step=False, on_epoch=True)
            #   # self.log('val_epoch_loss', self.epoch_loss['val'][-1], on_epoch=True, prog_bar=True, logger=True)
            self.step_outputs['val'].clear()


    def on_test_epoch_end(self) -> None:
        if self.global_rank==0:
            self.save_outputs(process="test")
            self.logger.log_metrics({'test_epoch_loss': self.epoch_loss['test'][-1]}, epoch=self.current_epoch)
            for metric_name in self.metrics_fns:
                self.logger.log_metrics({'test_epoch_{}'.format(metric_name): self.epoch_metrics['test'][metric_name][-1]}, step=self.current_epoch)
            # self.log('test_epoch_loss', self.epoch_loss['test'][-1], on_epoch=True, prog_bar=True, logger=True)
            self.step_outputs['test'].clear()
    
    
    #TODO: Implement on_predict_epoch_end/predict_epoch_end
    