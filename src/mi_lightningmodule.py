from typing import Any, Dict
from copy import deepcopy
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT

# from .mi_model import recurrent, AutoRegressiveTransformer
from .mi_eval import mi_mse, mi_smape, mi_pearsonr, mi_mae

class MILightningModule(pl.LightningModule):
    def __init__(self, args, model):
        super().__init__()
        
        # TODO: Separate model_args from trainer_args (May be)
        self.args = args
        # TODO: Move model instantiation outside LightningModule. 
        self.model = model
        
        self.metrics_fns = {}
        if args.num_classes>2:
            self.loss = nn.CrossEntropyLoss(weight=args.cross_entropy_class_weight)
        elif args.num_classes==2:
            self.loss = nn.BCEWithLogitsLoss()
        elif args.num_classes==1:
            self.loss = mi_smape #nn.MSELoss()
            # self.metrics_fns = {'mse': mi_mse}
            self.metrics_fns = {'smape': mi_smape, 'pearsonr': mi_pearsonr, 'mse': mi_mse}
        else:
            raise ValueError("Invalid number of classes")

        # Store step outputs
        self.step_outputs = dict(train=[], val=[], test=[])
        # Store avg loss over epochs
        self.epoch_loss = dict(train=[], val=[], test=[])
        # Store the metrics for each epoch for each of train, dev and test
        process_metric_dict = dict([(i, []) for i in self.metrics_fns])
        self.epoch_metrics = dict(train=deepcopy(process_metric_dict), val=deepcopy(process_metric_dict), test=deepcopy(process_metric_dict))
        # Store predictions and targets. Keys: preds, target since pytorch metrics use these arguments
        labels_dict = dict(preds=[], outcomes=[], outcomes_mask=[], infill_lang_mask=[])
        self.labels = dict(train=deepcopy(labels_dict), val=deepcopy(labels_dict), test=deepcopy(labels_dict))


    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
    
    
    def unpack_batch_model_inputs(self, batch):
        """
            Unpack the batch and return the model inputs separated from the other batch tensors
        """
        batch_labels = batch.pop('outcomes', None)
        query_id = batch.pop('query_ids', None)
        seq_id = batch.pop('seq_id', None) if 'seq_id' in batch else batch.pop('seq_idx', None)
        # rename 'infill_lang_mask' to 'mask'
        if 'infill_lang_mask' in batch: batch['mask'] = batch.pop('infill_lang_mask')
        return batch, batch_labels, seq_id
    
    
    def calculate_metrics(self, input:torch.Tensor, target:torch.Tensor, mask:torch.Tensor=None) -> Dict:
        metric_score = dict()
        for metric_name, fn in self.metrics_fns.items():
            metric_score[metric_name] = fn(input=input, target=target, mask=mask)
        return metric_score

    
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        batch, batch_labels, seq_id = self.unpack_batch_model_inputs(batch)
        batch_output = self.model(**batch)
        # Loss only calculates for the valid timesteps 
        batch_loss = self.loss(input=batch_output, target=batch_labels, mask=batch['outcomes_mask'])
        # if torch.isnan(batch_loss).any(): import pdb; pdb.set_trace()
        step_metrics = self.calculate_metrics(batch_output, batch_labels, batch['outcomes_mask'])
        log_metrics_dict = {'train_loss': batch_loss}
        log_metrics_dict.update({'train_{}'.format(key): val for key, val in step_metrics.items()})
        self.log_dict(log_metrics_dict, on_step=False, on_epoch=True)
        step_output = {'loss': batch_loss, 'pred': batch_output, 'outcomes': batch_labels, 'outcomes_mask': batch['outcomes_mask'], 
                    #    'infill_lang_mask': batch['mask'], 
                       'step': torch.tensor([self.global_step])}
        if seq_id is not None: step_output.update({'seq_id': seq_id})
        step_output.update(step_metrics)
        self.step_outputs['train'].append(step_output)
        return step_output
    
    
    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        batch, batch_labels, seq_id = self.unpack_batch_model_inputs(batch)
        batch_output = self.model(**batch)
        # TODO: Filter data based on three categories and compute loss/metrics for all categories:
        #                1. to the OOTS and Within Sample Sequence data
        #                2. to the OOTS and OOSS data
        #                3. to Within Time sample and OOSS data 
        #                4. to only OOTS data 
        #                5. to only OOSS data
        
        # Step 1: Get all data with ooss_mask = 0. Filter it down to oots_mask = 1. Compute within sequence - OOTS loss on this
        # Step 2: Get all data with ooss_mask = 1. Compute OOSS loss on this
        # Step 3: Get all data with ooss_mask = 1 and oots_mask = 0. Compute within time - OOSS loss on this
        # Step 4: Get all data with ooss_mask = 1 and oots_mask = 1. Compute OOTS-OOSS loss on this
        # Step 5: Get all data with oots_mask = 1. Compute OOTS loss on this
        
        ws_oots_batch_loss = torch.tensor([0.0])
        if torch.sum((batch['ooss_mask']==0) & (batch['oots_mask']==1))>0:
            batch_mask_subset = torch.where((batch['ooss_mask']==0).unsqueeze(-1) & (batch['oots_mask']==1).unsqueeze(-1), 
                                            batch['outcomes_mask'], torch.zeros_like(batch['outcomes_mask']))
            ws_oots_batch_loss = self.loss(input=batch_output, target=batch_labels, mask=batch_mask_subset)
            
        ooss_batch_loss = torch.tensor([0.0])
        if torch.sum(batch['ooss_mask']==1)>0:
            batch_mask_subset = torch.where((batch['ooss_mask']==1).unsqueeze(-1), batch['outcomes_mask'], 
                                            torch.zeros_like(batch['outcomes_mask']))
            ooss_batch_loss = self.loss(input=batch_output, target=batch_labels, mask=batch_mask_subset)
        
        wt_ooss_batch_loss = torch.tensor([0.0])
        if torch.sum((batch['ooss_mask']==1) & (batch['oots_mask']==0))>0:
            batch_mask_subset = torch.where((batch['ooss_mask']==1).unsqueeze(-1) & (batch['oots_mask']==0).unsqueeze(-1), 
                                            batch['outcomes_mask'], torch.zeros_like(batch['outcomes_mask']))
            wt_ooss_batch_loss = self.loss(input=batch_output, target=batch_labels, mask=batch_mask_subset)
        
        oots_ooss_batch_loss = torch.tensor([0.0])
        if torch.sum((batch['ooss_mask']==1) & (batch['oots_mask']==1))>0:
            batch_mask_subset = torch.where((batch['ooss_mask']==1).unsqueeze(-1) & (batch['oots_mask']==1).unsqueeze(-1), 
                                            batch['outcomes_mask'], torch.zeros_like(batch['outcomes_mask']))
            oots_ooss_batch_loss = self.loss(input=batch_output, target=batch_labels, mask=batch_mask_subset)
        
        oots_batch_loss = torch.tensor([0.0])
        if torch.sum(batch['oots_mask']==1)>0:
            batch_mask_subset = torch.where((batch['oots_mask']==1).unsqueeze(-1), batch['outcomes_mask'], 
                                            torch.zeros_like(batch['outcomes_mask']))
            oots_batch_loss = self.loss(input=batch_output, target=batch_labels, mask=batch_mask_subset)
        
        # batch_loss = self.loss(input=batch_output, target=batch_labels, mask=batch['outcomes_mask'])
        # step_metrics = self.calculate_metrics(batch_output, batch_labels, batch['outcomes_mask'])
        # log_metrics_dict = {'val_loss': batch_loss}
        # log_metrics_dict.update({'val_{}'.format(key): val for key, val in step_metrics.items()})
        # self.log_dict(log_metrics_dict, on_step=False, on_epoch=True)
        # step_output = {'loss': batch_loss, 'pred': batch_output, 'outcomes': batch_labels, 'outcomes_mask': batch['outcomes_mask'], 
        #             #    'infill_lang_mask': batch['mask'], 
        #                'step': torch.tensor([self.global_step])}
        # if seq_id is not None: step_output.update({'seq_id': seq_id})
        # step_output.update(step_metrics)
        # self.step_outputs['val'].append(step_output) 
        log_metrics_dict = {'val_ws_oots_loss': ws_oots_batch_loss, 'val_ooss_loss': ooss_batch_loss,
                            'val_wt_ooss_loss': wt_ooss_batch_loss, 'val_oots_ooss_loss': oots_ooss_batch_loss,
                            'val_oots_loss': oots_batch_loss}
        self.log_dict(log_metrics_dict, on_step=False, on_epoch=True)
        step_output = {'ws_oots_loss': ws_oots_batch_loss, 'ooss_loss': ooss_batch_loss,
                          'wt_ooss_loss': wt_ooss_batch_loss, 'oots_ooss_loss': oots_ooss_batch_loss,
                            'oots_loss': oots_batch_loss, 'pred': batch_output, 'outcomes': batch_labels,
                            'outcomes_mask': batch['outcomes_mask'], 'step': torch.tensor([self.global_step])}
        if seq_id is not None: step_output.update({'seq_id': seq_id})
        self.step_outputs['val'].append(step_output)

        return step_output
    
    
    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        batch, batch_labels, seq_id = self.unpack_batch_model_inputs(batch)
        batch_output = self.model(**batch)
        # TODO: Filter data based on two categories and compute loss/metrics for all categories:
        #               1. to the OOTS sample
        #               2. to the within time samples
        batch_loss = self.loss(input=batch_output, target=batch_labels)
        step_metrics = self.calculate_metrics(batch_output, batch_labels)
        log_metrics_dict = {'test_loss': batch_loss}
        log_metrics_dict.update({'test_{}'.format(key): val for key, val in step_metrics.items()})
        self.log_dict(log_metrics_dict, on_step=False, on_epoch=True)
        step_output = {'loss': batch_loss, 'pred': batch_output, 'outcomes': batch_labels, 
                    #    'infill_lang_mask': batch['mask'],
                       'outcomes_mask': batch['outcomes_mask'], }
        if seq_id is not None: step_output.update({'seq_id': seq_id})
        step_output.update(step_metrics)
        self.step_outputs['test'].append(step_output) 
        return step_output


    def predict_step(self, batch, batch_idx) -> STEP_OUTPUT:
        batch, batch_labels, seq_id = self.unpack_batch_model_inputs(batch)
        batch_output = self.model(**batch)
        return {'pred': batch_output, 'outcomes': batch_labels, 'seq_id': seq_id, 
                'outcomes_mask': batch['outcomes_mask'] }#'infill_lang_mask': batch['mask']}


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
                if step_output[k] is not None:
                    cat_outputs[k].append(step_output[k].detach().clone().cpu())
                else:
                    cat_outputs[k].append(None)
            
        return cat_outputs 
         

    def save_outputs(self, process:str) -> None:
        cat_outputs = self.process_epoch_end(self.step_outputs[process])
        
        if process == 'train':
            if 'loss' in cat_outputs:
                avg_loss = torch.tensor(cat_outputs['loss']).mean()
                self.epoch_loss[process].append(avg_loss.item())
        elif process == 'val':
            avg_ws_oots_loss = torch.tensor(cat_outputs['ws_oots_loss']).mean() if 'ws_oots_loss' in cat_outputs else None
            avg_ooss_loss = torch.tensor(cat_outputs['ooss_loss']).mean() if 'ooss_loss' in cat_outputs else None
            avg_wt_ooss_loss = torch.tensor(cat_outputs['wt_ooss_loss']).mean() if 'wt_ooss_loss' in cat_outputs else None
            avg_oots_ooss_loss = torch.tensor(cat_outputs['oots_ooss_loss']).mean() if 'oots_ooss_loss' in cat_outputs else None
            avg_oots_loss = torch.tensor(cat_outputs['oots_loss']).mean() if 'oots_loss' in cat_outputs else None
            self.epoch_loss[process].append({'ws_oots_loss': avg_ws_oots_loss, 'ooss_loss': avg_ooss_loss,
                                            'wt_ooss_loss': avg_wt_ooss_loss, 'oots_ooss_loss': avg_oots_ooss_loss,
                                            'oots_loss': avg_oots_loss})
        if 'pred' in cat_outputs:
            self.labels[process]['preds'].append(cat_outputs['pred'])
        if 'outcomes' in cat_outputs:
            self.labels[process]['outcomes'].append(cat_outputs['outcomes'])
        if 'outcomes_mask' in cat_outputs:
            self.labels[process]['outcomes_mask'].append(cat_outputs['outcomes_mask'])
        if 'infill_lang_mask' in cat_outputs:
            self.labels[process]['infill_lang_mask'].append(cat_outputs['infill_lang_mask'])
        # TODO: add ooss_mask and oots_mask to the labels 
        if self.metrics_fns:
            max_timesteps = max([i.shape[1] for i in cat_outputs['pred']])
            for batch_idx in range(len(cat_outputs['pred'])):
                if cat_outputs['pred'][batch_idx].shape[1] < max_timesteps:
                    zero_pads = torch.zeros((cat_outputs['pred'][batch_idx].shape[0], max_timesteps-cat_outputs['pred'][batch_idx].shape[1], self.args.num_outcomes))                    
                    cat_outputs['pred'][batch_idx] = torch.cat([cat_outputs['pred'][batch_idx], zero_pads], dim=1)
                    cat_outputs['outcomes'][batch_idx] = torch.cat([cat_outputs['outcomes'][batch_idx], zero_pads], dim=1)
                    cat_outputs['outcomes_mask'][batch_idx] = torch.cat([cat_outputs['outcomes_mask'][batch_idx], zero_pads], dim=1)
            pred_cat = torch.cat(cat_outputs['pred'], dim=0)
            target_cat = torch.cat(cat_outputs['outcomes'], dim=0)
            mask_cat = torch.cat(cat_outputs['outcomes_mask'], dim=0)
            for metric_name, fn in self.metrics_fns.items():
                value = fn(input=pred_cat, target=target_cat, mask=mask_cat)
                self.epoch_metrics[process][metric_name].append(value.item())


    def on_train_epoch_end(self) -> None:
        if self.global_rank==0:
            self.save_outputs(process="train")
            # TODO: We want epoch level flattened loss
            self.logger.log_metrics({'train_epoch_loss': self.epoch_loss['train'][-1]}, epoch=self.current_epoch)
            for metric_name in self.metrics_fns:
                self.logger.log_metrics({'train_epoch_{}'.format(metric_name): self.epoch_metrics['train'][metric_name][-1]}, epoch=self.current_epoch)
            self.step_outputs['train'].clear()

    
    def on_validation_epoch_end(self) -> None:
        if self.global_rank==0:
            self.save_outputs(process="val")
            self.logger.log_metrics({'val_epoch_ws_oots_loss': self.epoch_loss['val'][-1]['ws_oots_loss']}, epoch=self.current_epoch)
            self.logger.log_metrics({'val_epoch_ooss_loss': self.epoch_loss['val'][-1]['ooss_loss']}, epoch=self.current_epoch)
            self.logger.log_metrics({'val_epoch_wt_ooss_loss': self.epoch_loss['val'][-1]['wt_ooss_loss']}, epoch=self.current_epoch)
            self.logger.log_metrics({'val_epoch_oots_ooss_loss': self.epoch_loss['val'][-1]['oots_ooss_loss']}, epoch=self.current_epoch)
            self.logger.log_metrics({'val_epoch_oots_loss': self.epoch_loss['val'][-1]['oots_loss']}, epoch=self.current_epoch)
            # self.logger.log_metrics({'val_epoch_loss': self.epoch_loss['val'][-1]}, epoch=self.current_epoch)
            # for metric_name in self.metrics_fns:
            #     self.logger.log_metrics({'val_epoch_{}'.format(metric_name): self.epoch_metrics['val'][metric_name][-1]}, epoch=self.current_epoch)
            self.step_outputs['val'].clear()


    def on_test_epoch_end(self) -> None:
        if self.global_rank==0:
            self.save_outputs(process="test")
            self.logger.log_metrics({'test_epoch_loss': self.epoch_loss['test'][-1]}, epoch=self.current_epoch)
            for metric_name in self.metrics_fns:
                self.logger.log_metrics({'test_epoch_{}'.format(metric_name): self.epoch_metrics['test'][metric_name][-1]}, step=self.current_epoch)
            self.step_outputs['test'].clear() 