from typing import Any, Dict
from copy import deepcopy
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT

from .mi_eval import mi_mse, mi_smape, mi_pearsonr, mi_mae

class MILightningModule(pl.LightningModule):
    def __init__(self, args, model):
        super().__init__()
        
        # TODO: Separate model_args from trainer_args (May be)
        self.args = args
        self.model = model
        self.processor = TimeShiftProcessor(do_shift=args.do_shift, interpolation=args.interpolated_output)
        if args.max_scheduled_epochs == -1: args.max_scheduled_epochs = args.epochs
        if args.max_seq_len == -1: args.max_seq_len = args.max_len
        self.seq_len_scheduler = SequenceLengthScheduler(min_seq_len=args.min_seq_len, max_seq_len=args.max_seq_len, 
                                                         num_epochs=args.max_scheduled_epochs,
                                                         scheduler_type=args.seq_len_scheduler_type) if args.seq_len_scheduler_type != 'none' else None
                
        self.metrics_fns = {}
        if args.num_classes>2:
            self.loss = nn.CrossEntropyLoss(weight=args.cross_entropy_class_weight)
        elif args.num_classes==2:
            self.loss = nn.BCEWithLogitsLoss()
        elif args.num_classes==1:
            self.loss = mi_mse
            self.metrics_fns = {'smape': mi_smape, 'pearsonr': mi_pearsonr, 'mae': mi_mae, 'mse': mi_mse}
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
        labels_dict = dict(preds=[], outcomes=[], outcomes_mask=[], infill_lang_mask=[], seq_id=[], time_ids=[], oots_mask=[], ooss_mask=[])
        self.labels = dict(train=deepcopy(labels_dict), val=deepcopy(labels_dict), test=deepcopy(labels_dict))
        
        if args.custom_model == 'last_n_pcl_mean':
            # Important: This property activates manual optimization.
            self.automatic_optimization = False
            

    def configure_optimizers(self) -> Any:
        # return torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        return torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
    
    
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
            metric_score[metric_name] = fn(input=input, target=target, mask=mask, reduction=self.args.metrics_reduction)
        return metric_score

    
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        # TODO: CHECK IF ALL USERS HAVE ooss==0
        # TODO: CHECK IF ALL TIME POINTS HAVE oots==0
        batch, batch_labels, seq_id = self.unpack_batch_model_inputs(batch)
        if self.seq_len_scheduler: batch['outcomes_mask'] = self.seq_len_scheduler.trim_sequence(outcomes_mask=batch['outcomes_mask'],
                                                                      current_epoch=self.current_epoch)
        batch_output = self.model(**batch)
        batch_labels_diffed = self.processor.shift_labels(batch_labels)
        # Loss only calculates for the valid timesteps
        outcomes_mask = batch['outcomes_mask'] & (batch['oots_mask'].unsqueeze(-1) == 0) 
        batch_loss = self.loss(input=batch_output, target=batch_labels_diffed, mask=outcomes_mask, reduction=self.args.loss_reduction)
        
        batch_output_adj = self.processor.reshift_labels(batch_output, batch_labels, batch['outcomes_mask']) 
        step_metrics = self.calculate_metrics(batch_output_adj, batch_labels, batch['outcomes_mask'])
        log_metrics_dict = {'train_loss': batch_loss}
        log_metrics_dict.update({'train_{}'.format(key): val for key, val in step_metrics.items()})
        self.log_dict(log_metrics_dict, on_step=False, on_epoch=True)
        step_output = {'loss': batch_loss, 'pred': batch_output_adj, 'outcomes': batch_labels, 
                       'outcomes_mask': batch['outcomes_mask'], 'seq_id': seq_id, 'time_ids': batch['time_ids'],
                        'oots_mask': batch['oots_mask'], 'ooss_mask': batch['ooss_mask'],
                       'step': torch.tensor([self.global_step])}
        if seq_id is not None: step_output.update({'seq_id': seq_id})
        step_output.update(step_metrics)
        self.step_outputs['train'].append(step_output)
        
        if self.args.custom_model == 'last_n_pcl_mean':
            self.optimizers().zero_grad()
            # self.manual_backward(batch_loss)
            self.optimizers().step()
            
        return step_output
    
    
    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        batch, batch_labels, seq_id = self.unpack_batch_model_inputs(batch)
        batch_output = self.model(**batch)
        batch_labels_diffed = self.processor.shift_labels(batch_labels)
        
        batch_loss = self.loss(input=batch_output, target=batch_labels_diffed, mask=batch['outcomes_mask'], reduction=self.args.loss_reduction)
        batch_output_adj = self.processor.reshift_labels(batch_output, batch_labels, batch['outcomes_mask'])
        # Filter data based on five categories and compute loss/metrics for all categories:
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
        
        default_loss = torch.tensor([-1.0], device=batch_output.device)
        default_pearsonr = torch.tensor([-2.0], device=batch_output.device)
        default_smape = torch.tensor([-1.0], device=batch_output.device)
        default_mae = torch.tensor([-1.0], device=batch_output.device)
        default_mse = torch.tensor([-1.0], device=batch_output.device)
        
        ws_wt_batch_loss, ws_wt_pearsonr, ws_wt_smape = default_loss, default_pearsonr, default_smape
        ws_wt_mae, ws_wt_mse = default_mae, default_mse
        if torch.sum((batch['ooss_mask']==0) & (batch['oots_mask']==0))>0:
            batch_mask_subset = torch.where((batch['ooss_mask']==0).unsqueeze(-1) & (batch['oots_mask']==0).unsqueeze(-1),
                                            batch['outcomes_mask'], torch.zeros_like(batch['outcomes_mask']))
            batch_output_ = batch_output[torch.sum(batch_mask_subset, dim=[1, 2])>0]
            batch_output_adj_ = batch_output_adj[torch.sum(batch_mask_subset, dim=[1, 2])>0]
            batch_labels_diffed_ = batch_labels_diffed[torch.sum(batch_mask_subset, dim=[1, 2])>0]
            batch_labels_ = batch_labels[torch.sum(batch_mask_subset, dim=[1, 2])>0]
            batch_mask_subset = batch_mask_subset[torch.sum(batch_mask_subset, dim=[1, 2])>0]
            ws_wt_batch_loss = self.loss(input=batch_output_, target=batch_labels_diffed_, mask=batch_mask_subset, reduction=self.args.loss_reduction)
            ws_wt_pearsonr = mi_pearsonr(input=batch_output_adj_, target=batch_labels_, mask=batch_mask_subset, reduction=self.args.metrics_reduction)
            ws_wt_smape = mi_smape(input=batch_output_adj_, target=batch_labels_, mask=batch_mask_subset, reduction=self.args.metrics_reduction)
            ws_wt_mae = mi_mae(input=batch_output_adj_, target=batch_labels_, mask=batch_mask_subset, reduction=self.args.metrics_reduction)
            ws_wt_mse = mi_mse(input=batch_output_adj_, target=batch_labels_, mask=batch_mask_subset, reduction=self.args.metrics_reduction)
        
        valset_batch_loss, valset_pearsonr, valset_smape = default_loss, default_pearsonr, default_smape
        valset_mae, valset_mse = default_mae, default_mse
        # get all records that are not oots==0 and ooss==0
        if torch.sum((batch['ooss_mask']==1) | batch['oots_mask']==1)>0:
            batch_mask_subset = torch.where((batch['ooss_mask']==1).unsqueeze(-1) | (batch['oots_mask']==1).unsqueeze(-1), 
                                            batch['outcomes_mask'], torch.zeros_like(batch['outcomes_mask']))
            batch_output_ = batch_output[torch.sum(batch_mask_subset, dim=[1, 2])>0]
            batch_output_adj_ = batch_output_adj[torch.sum(batch_mask_subset, dim=[1, 2])>0]
            batch_labels_diffed_ = batch_labels_diffed[torch.sum(batch_mask_subset, dim=[1, 2])>0]
            batch_labels_ = batch_labels[torch.sum(batch_mask_subset, dim=[1, 2])>0]
            batch_mask_subset = batch_mask_subset[torch.sum(batch_mask_subset, dim=[1, 2])>0]
            valset_batch_loss = self.loss(input=batch_output_, target=batch_labels_diffed_, mask=batch_mask_subset, reduction=self.args.loss_reduction)
            valset_pearsonr = mi_pearsonr(input=batch_output_adj_, target=batch_labels_, mask=batch_mask_subset, reduction=self.args.metrics_reduction)
            valset_smape = mi_smape(input=batch_output_adj_, target=batch_labels_, mask=batch_mask_subset, reduction=self.args.metrics_reduction)
            valset_mae = mi_mae(input=batch_output_adj_, target=batch_labels_, mask=batch_mask_subset, reduction=self.args.metrics_reduction)
            valset_mse = mi_mse(input=batch_output_adj_, target=batch_labels_, mask=batch_mask_subset, reduction=self.args.metrics_reduction)

        ws_oots_batch_loss, ws_oots_pearsonr, ws_oots_smape = default_loss, default_pearsonr, default_smape
        ws_oots_mae, ws_oots_mse = default_mae, default_mse
        if torch.sum((batch['ooss_mask']==0) & (batch['oots_mask']==1))>0:
            batch_mask_subset = torch.where((batch['ooss_mask']==0).unsqueeze(-1) & (batch['oots_mask']==1).unsqueeze(-1), 
                                            batch['outcomes_mask'], torch.zeros_like(batch['outcomes_mask']))
            # import pdb; pdb.set_trace()
            # Remove the rows which have a sum of batch_mask across the sequence as 0 to avoid div by 0 error
            batch_output_ = batch_output[torch.sum(batch_mask_subset, dim=[1, 2])>0]
            batch_output_adj_ = batch_output_adj[torch.sum(batch_mask_subset, dim=[1, 2])>0]
            batch_labels_diffed_ = batch_labels_diffed[torch.sum(batch_mask_subset, dim=[1, 2])>0]
            batch_labels_ = batch_labels[torch.sum(batch_mask_subset, dim=[1, 2])>0]
            batch_mask_subset = batch_mask_subset[torch.sum(batch_mask_subset, dim=[1, 2])>0]
            ws_oots_batch_loss = self.loss(input=batch_output_, target=batch_labels_diffed_, mask=batch_mask_subset, reduction=self.args.loss_reduction)
            ws_oots_pearsonr = mi_pearsonr(input=batch_output_adj_, target=batch_labels_, mask=batch_mask_subset, reduction=self.args.metrics_reduction)
            ws_oots_smape = mi_smape(input=batch_output_adj_, target=batch_labels_, mask=batch_mask_subset, reduction=self.args.metrics_reduction)
            ws_oots_mae = mi_mae(input=batch_output_adj_, target=batch_labels_, mask=batch_mask_subset, reduction=self.args.metrics_reduction)
            ws_oots_mse = mi_mse(input=batch_output_adj_, target=batch_labels_, mask=batch_mask_subset, reduction=self.args.metrics_reduction)
            
        wt_ooss_batch_loss, wt_ooss_pearsonr, wt_ooss_smape = default_loss, default_pearsonr, default_smape
        wt_ooss_mae, wt_ooss_mse = default_mae, default_mse
        if torch.sum((batch['ooss_mask']==1) & (batch['oots_mask']==0))>0:
            batch_mask_subset = torch.where((batch['ooss_mask']==1).unsqueeze(-1) & (batch['oots_mask']==0).unsqueeze(-1), 
                                            batch['outcomes_mask'], torch.zeros_like(batch['outcomes_mask']))
            batch_output_ = batch_output[torch.sum(batch_mask_subset, dim=[1, 2])>0]
            batch_output_adj_ = batch_output_adj[torch.sum(batch_mask_subset, dim=[1, 2])>0]
            batch_labels_diffed_ = batch_labels_diffed[torch.sum(batch_mask_subset, dim=[1, 2])>0]
            batch_labels_ = batch_labels[torch.sum(batch_mask_subset, dim=[1, 2])>0]
            batch_mask_subset = batch_mask_subset[torch.sum(batch_mask_subset, dim=[1, 2])>0]
            wt_ooss_batch_loss = self.loss(input=batch_output_, target=batch_labels_diffed_, mask=batch_mask_subset, reduction=self.args.loss_reduction)
            wt_ooss_pearsonr = mi_pearsonr(input=batch_output_adj_, target=batch_labels_, mask=batch_mask_subset, reduction=self.args.metrics_reduction)
            wt_ooss_smape = mi_smape(input=batch_output_adj_, target=batch_labels_, mask=batch_mask_subset, reduction=self.args.metrics_reduction)
            wt_ooss_mae = mi_mae(input=batch_output_adj_, target=batch_labels_, mask=batch_mask_subset, reduction=self.args.metrics_reduction)
            wt_ooss_mse = mi_mse(input=batch_output_adj_, target=batch_labels_, mask=batch_mask_subset, reduction=self.args.metrics_reduction)
            
        oots_ooss_batch_loss, oots_ooss_pearsonr, oots_ooss_smape = default_loss, default_pearsonr, default_smape
        oots_ooss_mae, oots_ooss_mse = default_mae, default_mse
        if torch.sum((batch['ooss_mask']==1) & (batch['oots_mask']==1))>0:
            batch_mask_subset = torch.where((batch['ooss_mask']==1).unsqueeze(-1) & (batch['oots_mask']==1).unsqueeze(-1), 
                                            batch['outcomes_mask'], torch.zeros_like(batch['outcomes_mask']))
            batch_output_ = batch_output[torch.sum(batch_mask_subset, dim=[1, 2])>0]
            batch_output_adj_ = batch_output_adj[torch.sum(batch_mask_subset, dim=[1, 2])>0]
            batch_labels_diffed_ = batch_labels_diffed[torch.sum(batch_mask_subset, dim=[1, 2])>0]
            batch_labels_ = batch_labels[torch.sum(batch_mask_subset, dim=[1, 2])>0]
            batch_mask_subset = batch_mask_subset[torch.sum(batch_mask_subset, dim=[1, 2])>0]
            oots_ooss_batch_loss = self.loss(input=batch_output_, target=batch_labels_diffed_, mask=batch_mask_subset, reduction=self.args.loss_reduction)
            oots_ooss_pearsonr = mi_pearsonr(input=batch_output_adj_, target=batch_labels_, mask=batch_mask_subset, reduction=self.args.metrics_reduction)
            oots_ooss_smape = mi_smape(input=batch_output_adj_, target=batch_labels_, mask=batch_mask_subset, reduction=self.args.metrics_reduction)
            oots_ooss_mae = mi_mae(input=batch_output_adj_, target=batch_labels_, mask=batch_mask_subset, reduction=self.args.metrics_reduction)
            oots_ooss_mse = mi_mse(input=batch_output_adj_, target=batch_labels_, mask=batch_mask_subset, reduction=self.args.metrics_reduction)
            
        oots_batch_loss, oots_pearsonr, oots_smape = default_loss, default_pearsonr, default_smape
        oots_mae, oots_mse = default_mae, default_mse
        if torch.sum(batch['oots_mask']==1)>0:
            batch_mask_subset = torch.where((batch['oots_mask']==1).unsqueeze(-1), batch['outcomes_mask'], 
                                            torch.zeros_like(batch['outcomes_mask']))
            batch_output_ = batch_output[torch.sum(batch_mask_subset, dim=[1, 2])>0]
            batch_output_adj_ = batch_output_adj[torch.sum(batch_mask_subset, dim=[1, 2])>0]
            batch_labels_diffed_ = batch_labels_diffed[torch.sum(batch_mask_subset, dim=[1, 2])>0]
            batch_labels_ = batch_labels[torch.sum(batch_mask_subset, dim=[1, 2])>0]
            batch_mask_subset = batch_mask_subset[torch.sum(batch_mask_subset, dim=[1, 2])>0]
            oots_batch_loss = self.loss(input=batch_output_, target=batch_labels_diffed_, mask=batch_mask_subset, reduction=self.args.loss_reduction)
            oots_pearsonr = mi_pearsonr(input=batch_output_adj_, target=batch_labels_, mask=batch_mask_subset, reduction=self.args.metrics_reduction)
            oots_smape = mi_smape(input=batch_output_adj_, target=batch_labels_, mask=batch_mask_subset, reduction=self.args.metrics_reduction)
            oots_mae = mi_mae(input=batch_output_adj_, target=batch_labels_, mask=batch_mask_subset, reduction=self.args.metrics_reduction)
            oots_mse = mi_mse(input=batch_output_adj_, target=batch_labels_, mask=batch_mask_subset, reduction=self.args.metrics_reduction)
            
        ooss_batch_loss, ooss_pearsonr, ooss_smape = default_loss, default_pearsonr, default_smape
        ooss_mae, ooss_mse = default_mae, default_mse
        if torch.sum(batch['ooss_mask']==1)>0:
            batch_mask_subset = torch.where((batch['ooss_mask']==1).unsqueeze(-1), batch['outcomes_mask'], 
                                            torch.zeros_like(batch['outcomes_mask']))
            batch_output_ = batch_output[torch.sum(batch_mask_subset, dim=[1, 2])>0]
            batch_output_adj_ = batch_output_adj[torch.sum(batch_mask_subset, dim=[1, 2])>0]
            batch_labels_diffed_ = batch_labels_diffed[torch.sum(batch_mask_subset, dim=[1, 2])>0]
            batch_labels_ = batch_labels[torch.sum(batch_mask_subset, dim=[1, 2])>0]
            batch_mask_subset = batch_mask_subset[torch.sum(batch_mask_subset, dim=[1, 2])>0]
            ooss_batch_loss = self.loss(input=batch_output_, target=batch_labels_diffed_, mask=batch_mask_subset, reduction=self.args.loss_reduction)
            ooss_pearsonr = mi_pearsonr(input=batch_output_adj_, target=batch_labels_, mask=batch_mask_subset, reduction=self.args.metrics_reduction)
            ooss_smape = mi_smape(input=batch_output_adj_, target=batch_labels_, mask=batch_mask_subset, reduction=self.args.metrics_reduction)
            ooss_mae = mi_mae(input=batch_output_adj_, target=batch_labels_, mask=batch_mask_subset, reduction=self.args.metrics_reduction)
            ooss_mse = mi_mse(input=batch_output_adj_, target=batch_labels_, mask=batch_mask_subset, reduction=self.args.metrics_reduction)
            
        log_metrics_dict = {}        
        if not (ws_wt_batch_loss==default_loss): log_metrics_dict['val_ws_wt_loss'] = ws_wt_batch_loss
        if not (valset_batch_loss==default_loss): log_metrics_dict['val_valset_loss'] = valset_batch_loss
        if not (ws_oots_batch_loss==default_loss): log_metrics_dict['val_ws_oots_loss'] = ws_oots_batch_loss
        if not (wt_ooss_batch_loss==default_loss): log_metrics_dict['val_wt_ooss_loss'] = wt_ooss_batch_loss
        if not (oots_ooss_batch_loss==default_loss): log_metrics_dict['val_oots_ooss_loss'] = oots_ooss_batch_loss
        if not (oots_batch_loss==default_loss): log_metrics_dict['val_oots_loss'] = oots_batch_loss
        if not (ooss_batch_loss==default_loss): log_metrics_dict['val_ooss_loss'] = ooss_batch_loss

        if not (ws_wt_pearsonr==default_pearsonr): log_metrics_dict['val_ws_wt_pearsonr'] = ws_wt_pearsonr
        if not (valset_pearsonr==default_pearsonr): log_metrics_dict['val_valset_pearsonr'] = valset_pearsonr
        if not (ws_oots_pearsonr==default_pearsonr): log_metrics_dict['val_ws_oots_pearsonr'] = ws_oots_pearsonr
        if not (wt_ooss_pearsonr==default_pearsonr): log_metrics_dict['val_wt_ooss_pearsonr'] = wt_ooss_pearsonr
        if not (oots_ooss_pearsonr==default_pearsonr): log_metrics_dict['val_oots_ooss_pearsonr'] = oots_ooss_pearsonr
        if not (oots_pearsonr==default_pearsonr): log_metrics_dict['val_oots_pearsonr'] = oots_pearsonr
        if not (ooss_pearsonr==default_pearsonr): log_metrics_dict['val_ooss_pearsonr'] = ooss_pearsonr

        if not (ws_wt_smape==default_smape): log_metrics_dict['val_ws_wt_smape'] = ws_wt_smape
        if not (valset_smape==default_smape): log_metrics_dict['val_valset_smape'] = valset_smape
        if not (ws_oots_smape==default_smape): log_metrics_dict['val_ws_oots_smape'] = ws_oots_smape
        if not (wt_ooss_smape==default_smape): log_metrics_dict['val_wt_ooss_smape'] = wt_ooss_smape
        if not (oots_ooss_smape==default_smape): log_metrics_dict['val_oots_ooss_smape'] = oots_ooss_smape
        if not (oots_smape==default_smape): log_metrics_dict['val_oots_smape'] = oots_smape
        if not (ooss_smape==default_smape): log_metrics_dict['val_ooss_smape'] = ooss_smape

        if not (ws_wt_mae==default_mae): log_metrics_dict['val_ws_wt_mae'] = ws_wt_mae
        if not (valset_mae==default_mae): log_metrics_dict['val_valset_mae'] = valset_mae
        if not (ws_oots_mae==default_mae): log_metrics_dict['val_ws_oots_mae'] = ws_oots_mae
        if not (wt_ooss_mae==default_mae): log_metrics_dict['val_wt_ooss_mae'] = wt_ooss_mae
        if not (oots_ooss_mae==default_mae): log_metrics_dict['val_oots_ooss_mae'] = oots_ooss_mae
        if not (oots_mae==default_mae): log_metrics_dict['val_oots_mae'] = oots_mae
        if not (ooss_mae==default_mae): log_metrics_dict['val_ooss_mae'] = ooss_mae

        if not (ws_wt_mse==default_mse): log_metrics_dict['val_ws_wt_mse'] = ws_wt_mse
        if not (valset_mse==default_mse): log_metrics_dict['val_valset_mse'] = valset_mse
        if not (ws_oots_mse==default_mse): log_metrics_dict['val_ws_oots_mse'] = ws_oots_mse
        if not (wt_ooss_mse==default_mse): log_metrics_dict['val_wt_ooss_mse'] = wt_ooss_mse
        if not (oots_ooss_mse==default_mse): log_metrics_dict['val_oots_ooss_mse'] = oots_ooss_mse
        if not (oots_mse==default_mse): log_metrics_dict['val_oots_mse'] = oots_mse
        if not (ooss_mse==default_mse): log_metrics_dict['val_ooss_mse'] = ooss_mse
        
        self.log_dict(log_metrics_dict, on_step=False, on_epoch=True)
        step_output = {'ws_wt_loss': ws_wt_batch_loss, 'valset_loss': valset_batch_loss, 'ws_oots_loss': ws_oots_batch_loss, 'wt_ooss_loss': wt_ooss_batch_loss, 
                       'oots_ooss_loss': oots_ooss_batch_loss, 'oots_loss': oots_batch_loss, 'ooss_loss': ooss_batch_loss,  
                        'ws_wt_pearsonr': ws_wt_pearsonr, 'valset_pearsonr': valset_pearsonr, 'ws_oots_pearsonr': ws_oots_pearsonr, 'wt_ooss_pearsonr': wt_ooss_pearsonr, 
                        'oots_ooss_pearsonr': oots_ooss_pearsonr, 'oots_pearsonr': oots_pearsonr, 'ooss_pearsonr': ooss_pearsonr, 
                        'ws_wt_smape': ws_wt_smape, 'valset_smape': valset_smape, 'ws_oots_smape': ws_oots_smape, 'wt_ooss_smape': wt_ooss_smape, 
                        'oots_ooss_smape': oots_ooss_smape, 'oots_smape': oots_smape, 'ooss_smape': ooss_smape,
                        'ws_wt_mae': ws_wt_mae, 'valset_mae': valset_mae, 'ws_oots_mae': ws_oots_mae, 'wt_ooss_mae': wt_ooss_mae, 'oots_ooss_mae': oots_ooss_mae,
                        'oots_mae': oots_mae, 'ooss_mae': ooss_mae,
                        'ws_wt_mse': ws_wt_mse, 'valset_mse': valset_mse, 'ws_oots_mse': ws_oots_mse, 'wt_ooss_mse': wt_ooss_mse, 'oots_ooss_mse': oots_ooss_mse,
                        'oots_mse': oots_mse, 'ooss_mse': ooss_mse,
                        'outcomes': batch_labels, 'outcomes_mask': batch['outcomes_mask'], 'step': torch.tensor([self.global_step]),
                        'seq_id': seq_id, 'ooss_mask': batch['ooss_mask'], 'oots_mask': batch['oots_mask'], 'pred': batch_output_adj,
                        'time_ids': batch['time_ids']}
        if seq_id is not None: step_output.update({'seq_id': seq_id})
        self.step_outputs['val'].append(step_output)

        return step_output
    
    
    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        batch, batch_labels, seq_id = self.unpack_batch_model_inputs(batch)
        batch_output = self.model(**batch)
        batch_labels_diffed = self.processor.shift_labels(batch_labels)
        batch_output_adj = self.processor.reshift_labels(batch_output, batch_labels, batch['outcomes_mask']) 
        # TODO: Filter data based on two categories and compute loss/metrics for all categories:
        #               1. to the OOTS sample
        #               2. to the within time samples
        batch_loss = self.loss(input=batch_output, target=batch_labels_diffed, mask=batch['outcomes_mask'], reduction=self.args.loss_reduction)
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
            avg_ws_wt_loss = torch.tensor([x for x in cat_outputs['ws_wt_loss'] if x >= 0]).mean() if 'ws_wt_loss' in cat_outputs else None
            avg_valset_loss = torch.tensor([x for x in cat_outputs['valset_loss'] if x >= 0]).mean() if 'valset_loss' in cat_outputs else None
            avg_ws_oots_loss = torch.tensor([x for x in cat_outputs['ws_oots_loss'] if x >= 0]).mean() if 'ws_oots_loss' in cat_outputs else None
            avg_wt_ooss_loss = torch.tensor([x for x in cat_outputs['wt_ooss_loss'] if x >= 0]).mean() if 'wt_ooss_loss' in cat_outputs else None
            avg_oots_ooss_loss = torch.tensor([x for x in cat_outputs['oots_ooss_loss'] if x >= 0]).mean() if 'oots_ooss_loss' in cat_outputs else None
            avg_oots_loss = torch.tensor([x for x in cat_outputs['oots_loss'] if x >= 0]).mean() if 'oots_loss' in cat_outputs else None
            avg_ooss_loss = torch.tensor([x for x in cat_outputs['ooss_loss'] if x >= 0]).mean() if 'ooss_loss' in cat_outputs else None
            avg_ws_wt_pearsonr = torch.tensor([x for x in cat_outputs['ws_wt_pearsonr'] if x >= -1]).mean() if 'ws_wt_pearsonr' in cat_outputs else None
            avg_valset_pearsonr = torch.tensor([x for x in cat_outputs['valset_pearsonr'] if x >= -1]).mean() if 'valset_pearsonr' in cat_outputs else None
            avg_ws_oots_pearsonr = torch.tensor([x for x in cat_outputs['ws_oots_pearsonr'] if x >= -1]).mean() if 'ws_oots_pearsonr' in cat_outputs else None
            avg_wt_ooss_pearsonr = torch.tensor([x for x in cat_outputs['wt_ooss_pearsonr'] if x >= -1]).mean() if 'wt_ooss_pearsonr' in cat_outputs else None
            avg_oots_ooss_pearsonr = torch.tensor([x for x in cat_outputs['oots_ooss_pearsonr'] if x >= -1]).mean() if 'oots_ooss_pearsonr' in cat_outputs else None
            avg_oots_pearsonr = torch.tensor([x for x in cat_outputs['oots_pearsonr'] if x >= -1]).mean() if 'oots_pearsonr' in cat_outputs else None
            avg_ooss_pearsonr = torch.tensor([x for x in cat_outputs['ooss_pearsonr'] if x >= -1]).mean() if 'ooss_pearsonr' in cat_outputs else None
            avg_ws_wt_smape = torch.tensor([x for x in cat_outputs['ws_wt_smape'] if x >= 0]).mean() if 'ws_wt_smape' in cat_outputs else None
            avg_valset_smape = torch.tensor([x for x in cat_outputs['valset_smape'] if x >= 0]).mean() if 'valset_smape' in cat_outputs else None
            avg_ws_oots_smape = torch.tensor([x for x in cat_outputs['ws_oots_smape'] if x >= 0]).mean() if 'ws_oots_smape' in cat_outputs else None
            avg_wt_ooss_smape = torch.tensor([x for x in cat_outputs['wt_ooss_smape'] if x >= 0]).mean() if 'wt_ooss_smape' in cat_outputs else None
            avg_oots_ooss_smape = torch.tensor([x for x in cat_outputs['oots_ooss_smape'] if x >= 0]).mean() if 'oots_ooss_smape' in cat_outputs else None
            avg_oots_smape = torch.tensor([x for x in cat_outputs['oots_smape'] if x >= 0]).mean() if 'oots_smape' in cat_outputs else None
            avg_ooss_smape = torch.tensor([x for x in cat_outputs['ooss_smape'] if x >= 0]).mean() if 'ooss_smape' in cat_outputs else None
            avg_ws_wt_mae = torch.tensor([x for x in cat_outputs['ws_wt_mae'] if x >= 0]).mean() if 'ws_wt_mae' in cat_outputs else None
            avg_valset_mae = torch.tensor([x for x in cat_outputs['valset_mae'] if x >= 0]).mean() if 'valset_mae' in cat_outputs else None
            avg_ws_oots_mae = torch.tensor([x for x in cat_outputs['ws_oots_mae'] if x >= 0]).mean() if 'ws_oots_mae' in cat_outputs else None
            avg_wt_ooss_mae = torch.tensor([x for x in cat_outputs['wt_ooss_mae'] if x >= 0]).mean() if 'wt_ooss_mae' in cat_outputs else None
            avg_oots_ooss_mae = torch.tensor([x for x in cat_outputs['oots_ooss_mae'] if x >= 0]).mean() if 'oots_ooss_mae' in cat_outputs else None
            avg_oots_mae = torch.tensor([x for x in cat_outputs['oots_mae'] if x >= 0]).mean() if 'oots_mae' in cat_outputs else None
            avg_ooss_mae = torch.tensor([x for x in cat_outputs['ooss_mae'] if x >= 0]).mean() if 'ooss_mae' in cat_outputs else None
            avg_ws_wt_mse = torch.tensor([x for x in cat_outputs['ws_wt_mse'] if x >= 0]).mean() if 'ws_wt_mse' in cat_outputs else None
            avg_valset_mse = torch.tensor([x for x in cat_outputs['valset_mse'] if x >= 0]).mean() if 'valset_mse' in cat_outputs else None
            avg_ws_oots_mse = torch.tensor([x for x in cat_outputs['ws_oots_mse'] if x >= 0]).mean() if 'ws_oots_mse' in cat_outputs else None
            avg_wt_ooss_mse = torch.tensor([x for x in cat_outputs['wt_ooss_mse'] if x >= 0]).mean() if 'wt_ooss_mse' in cat_outputs else None
            avg_oots_ooss_mse = torch.tensor([x for x in cat_outputs['oots_ooss_mse'] if x >= 0]).mean() if 'oots_ooss_mse' in cat_outputs else None
            avg_oots_mse = torch.tensor([x for x in cat_outputs['oots_mse'] if x >= 0]).mean() if 'oots_mse' in cat_outputs else None
            avg_ooss_mse = torch.tensor([x for x in cat_outputs['ooss_mse'] if x >= 0]).mean() if 'ooss_mse' in cat_outputs else None

            self.epoch_loss[process].append({'ws_wt_loss': avg_ws_wt_loss, 'valset_loss': avg_valset_loss,  'ws_oots_loss': avg_ws_oots_loss, 'wt_ooss_loss': avg_wt_ooss_loss,
                                            'oots_ooss_loss': avg_oots_ooss_loss, 'oots_loss': avg_oots_loss, 'ooss_loss': avg_ooss_loss,
                                            'ws_wt_pearsonr': avg_ws_wt_pearsonr, 'valset_pearsonr':avg_valset_pearsonr, 'ws_oots_pearsonr': avg_ws_oots_pearsonr, 'wt_ooss_pearsonr': avg_wt_ooss_pearsonr, 
                                            'oots_ooss_pearsonr': avg_oots_ooss_pearsonr, 'oots_pearsonr': avg_oots_pearsonr, 'ooss_pearsonr': avg_ooss_pearsonr,
                                            'ws_wt_smape': avg_ws_wt_smape, 'valset_smape': avg_valset_smape , 'ws_oots_smape': avg_ws_oots_smape, 'wt_ooss_smape': avg_wt_ooss_smape, 
                                            'oots_ooss_smape': avg_oots_ooss_smape, 'oots_smape': avg_oots_smape, 'ooss_smape': avg_ooss_smape,
                                            'ws_wt_mae': avg_ws_wt_mae, 'valset_mae': avg_valset_mae, 'ws_oots_mae': avg_ws_oots_mae, 'wt_ooss_mae': avg_wt_ooss_mae, 
                                            'oots_ooss_mae': avg_oots_ooss_mae, 'oots_mae': avg_oots_mae, 'ooss_mae': avg_ooss_mae,
                                            'ws_wt_mse': avg_ws_wt_mse, 'valset_mse': avg_valset_mse, 'ws_oots_mse': avg_ws_oots_mse, 'wt_ooss_mse': avg_wt_ooss_mse, 
                                            'oots_ooss_mse': avg_oots_ooss_mse, 'oots_mse': avg_oots_mse, 'ooss_mse': avg_ooss_mse})
        
        if 'pred' in cat_outputs: self.labels[process]['preds'].append(cat_outputs['pred'])
        if 'outcomes' in cat_outputs: self.labels[process]['outcomes'].append(cat_outputs['outcomes'])
        if 'outcomes_mask' in cat_outputs: self.labels[process]['outcomes_mask'].append(cat_outputs['outcomes_mask'])
        if 'infill_lang_mask' in cat_outputs: self.labels[process]['infill_lang_mask'].append(cat_outputs['infill_lang_mask'])
        if 'seq_id' in cat_outputs: self.labels[process]['seq_id'].append(cat_outputs['seq_id'])
        if 'time_ids' in cat_outputs: self.labels[process]['time_ids'].append(cat_outputs['time_ids'])
        if 'orig_time_ids' in cat_outputs: self.labels[process]['orig_time_ids'].append(cat_outputs['orig_time_ids'])
        if 'oots_mask' in cat_outputs: self.labels[process]['oots_mask'].append(cat_outputs['oots_mask'])
        if 'ooss_mask' in cat_outputs: self.labels[process]['ooss_mask'].append(cat_outputs['ooss_mask'])
        
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
            if ~torch.isnan(self.epoch_loss['val'][-1]['ws_wt_loss']): self.logger.log_metrics({'val_epoch_ws_wt_loss': self.epoch_loss['val'][-1]['ws_wt_loss']}, epoch=self.current_epoch)
            if ~torch.isnan(self.epoch_loss['val'][-1]['valset_loss']): self.logger.log_metrics({'val_epoch_valset_loss': self.epoch_loss['val'][-1]['valset_loss']}, epoch=self.current_epoch)
            if ~torch.isnan(self.epoch_loss['val'][-1]['ws_oots_loss']): self.logger.log_metrics({'val_epoch_ws_oots_loss': self.epoch_loss['val'][-1]['ws_oots_loss']}, epoch=self.current_epoch)
            if ~torch.isnan(self.epoch_loss['val'][-1]['wt_ooss_loss']): self.logger.log_metrics({'val_epoch_wt_ooss_loss': self.epoch_loss['val'][-1]['wt_ooss_loss']}, epoch=self.current_epoch)
            if ~torch.isnan(self.epoch_loss['val'][-1]['oots_ooss_loss']): self.logger.log_metrics({'val_epoch_oots_ooss_loss': self.epoch_loss['val'][-1]['oots_ooss_loss']}, epoch=self.current_epoch)
            if ~torch.isnan(self.epoch_loss['val'][-1]['oots_loss']): self.logger.log_metrics({'val_epoch_oots_loss': self.epoch_loss['val'][-1]['oots_loss']}, epoch=self.current_epoch)
            if ~torch.isnan(self.epoch_loss['val'][-1]['ooss_loss']): self.logger.log_metrics({'val_epoch_ooss_loss': self.epoch_loss['val'][-1]['ooss_loss']}, epoch=self.current_epoch)
            if ~torch.isnan(self.epoch_loss['val'][-1]['ws_wt_pearsonr']): self.logger.log_metrics({'val_epoch_ws_wt_pearsonr': self.epoch_loss['val'][-1]['ws_wt_pearsonr']}, epoch=self.current_epoch)
            if ~torch.isnan(self.epoch_loss['val'][-1]['valset_pearsonr']): self.logger.log_metrics({'val_epoch_valset_pearsonr': self.epoch_loss['val'][-1]['valset_pearsonr']}, epoch=self.current_epoch)
            if ~torch.isnan(self.epoch_loss['val'][-1]['ws_oots_pearsonr']): self.logger.log_metrics({'val_epoch_ws_oots_pearsonr': self.epoch_loss['val'][-1]['ws_oots_pearsonr']}, epoch=self.current_epoch)
            if ~torch.isnan(self.epoch_loss['val'][-1]['wt_ooss_pearsonr']): self.logger.log_metrics({'val_epoch_wt_ooss_pearsonr': self.epoch_loss['val'][-1]['wt_ooss_pearsonr']}, epoch=self.current_epoch)
            if ~torch.isnan(self.epoch_loss['val'][-1]['oots_ooss_pearsonr']): self.logger.log_metrics({'val_epoch_oots_ooss_pearsonr': self.epoch_loss['val'][-1]['oots_ooss_pearsonr']}, epoch=self.current_epoch)
            if ~torch.isnan(self.epoch_loss['val'][-1]['oots_pearsonr']): self.logger.log_metrics({'val_epoch_oots_pearsonr': self.epoch_loss['val'][-1]['oots_pearsonr']}, epoch=self.current_epoch)
            if ~torch.isnan(self.epoch_loss['val'][-1]['ooss_pearsonr']): self.logger.log_metrics({'val_epoch_ooss_pearsonr': self.epoch_loss['val'][-1]['ooss_pearsonr']}, epoch=self.current_epoch)
            if ~torch.isnan(self.epoch_loss['val'][-1]['ws_wt_smape']): self.logger.log_metrics({'val_epoch_ws_wt_smape': self.epoch_loss['val'][-1]['ws_wt_smape']}, epoch=self.current_epoch)
            if ~torch.isnan(self.epoch_loss['val'][-1]['valset_smape']): self.logger.log_metrics({'val_epoch_valset_smape': self.epoch_loss['val'][-1]['valset_smape']}, epoch=self.current_epoch)
            if ~torch.isnan(self.epoch_loss['val'][-1]['ws_oots_smape']): self.logger.log_metrics({'val_epoch_ws_oots_smape': self.epoch_loss['val'][-1]['ws_oots_smape']}, epoch=self.current_epoch)
            if ~torch.isnan(self.epoch_loss['val'][-1]['wt_ooss_smape']): self.logger.log_metrics({'val_epoch_wt_ooss_smape': self.epoch_loss['val'][-1]['wt_ooss_smape']}, epoch=self.current_epoch)
            if ~torch.isnan(self.epoch_loss['val'][-1]['oots_ooss_smape']): self.logger.log_metrics({'val_epoch_oots_ooss_smape': self.epoch_loss['val'][-1]['oots_ooss_smape']}, epoch=self.current_epoch)
            if ~torch.isnan(self.epoch_loss['val'][-1]['oots_smape']): self.logger.log_metrics({'val_epoch_oots_smape': self.epoch_loss['val'][-1]['oots_smape']}, epoch=self.current_epoch)
            if ~torch.isnan(self.epoch_loss['val'][-1]['ooss_smape']): self.logger.log_metrics({'val_epoch_ooss_smape': self.epoch_loss['val'][-1]['ooss_smape']}, epoch=self.current_epoch)
            if ~torch.isnan(self.epoch_loss['val'][-1]['ws_wt_mae']): self.logger.log_metrics({'val_epoch_ws_wt_mae': self.epoch_loss['val'][-1]['ws_wt_mae']}, epoch=self.current_epoch)
            if ~torch.isnan(self.epoch_loss['val'][-1]['valset_mae']): self.logger.log_metrics({'val_epoch_valset_mae': self.epoch_loss['val'][-1]['valset_mae']}, epoch=self.current_epoch)
            if ~torch.isnan(self.epoch_loss['val'][-1]['ws_oots_mae']): self.logger.log_metrics({'val_epoch_ws_oots_mae': self.epoch_loss['val'][-1]['ws_oots_mae']}, epoch=self.current_epoch)
            if ~torch.isnan(self.epoch_loss['val'][-1]['wt_ooss_mae']): self.logger.log_metrics({'val_epoch_wt_ooss_mae': self.epoch_loss['val'][-1]['wt_ooss_mae']}, epoch=self.current_epoch)
            if ~torch.isnan(self.epoch_loss['val'][-1]['oots_ooss_mae']): self.logger.log_metrics({'val_epoch_oots_ooss_mae': self.epoch_loss['val'][-1]['oots_ooss_mae']}, epoch=self.current_epoch)
            if ~torch.isnan(self.epoch_loss['val'][-1]['oots_mae']): self.logger.log_metrics({'val_epoch_oots_mae': self.epoch_loss['val'][-1]['oots_mae']}, epoch=self.current_epoch)
            if ~torch.isnan(self.epoch_loss['val'][-1]['ooss_mae']): self.logger.log_metrics({'val_epoch_ooss_mae': self.epoch_loss['val'][-1]['ooss_mae']}, epoch=self.current_epoch)
            if ~torch.isnan(self.epoch_loss['val'][-1]['ws_wt_mse']): self.logger.log_metrics({'val_epoch_ws_wt_mse': self.epoch_loss['val'][-1]['ws_wt_mse']}, epoch=self.current_epoch)
            if ~torch.isnan(self.epoch_loss['val'][-1]['valset_mse']): self.logger.log_metrics({'val_epoch_valset_mse': self.epoch_loss['val'][-1]['valset_mse']}, epoch=self.current_epoch)
            if ~torch.isnan(self.epoch_loss['val'][-1]['ws_oots_mse']): self.logger.log_metrics({'val_epoch_ws_oots_mse': self.epoch_loss['val'][-1]['ws_oots_mse']}, epoch=self.current_epoch)
            if ~torch.isnan(self.epoch_loss['val'][-1]['wt_ooss_mse']): self.logger.log_metrics({'val_epoch_wt_ooss_mse': self.epoch_loss['val'][-1]['wt_ooss_mse']}, epoch=self.current_epoch)
            if ~torch.isnan(self.epoch_loss['val'][-1]['oots_ooss_mse']): self.logger.log_metrics({'val_epoch_oots_ooss_mse': self.epoch_loss['val'][-1]['oots_ooss_mse']}, epoch=self.current_epoch)
            if ~torch.isnan(self.epoch_loss['val'][-1]['oots_mse']): self.logger.log_metrics({'val_epoch_oots_mse': self.epoch_loss['val'][-1]['oots_mse']}, epoch=self.current_epoch)
            if ~torch.isnan(self.epoch_loss['val'][-1]['ooss_mse']): self.logger.log_metrics({'val_epoch_ooss_mse': self.epoch_loss['val'][-1]['ooss_mse']}, epoch=self.current_epoch)
            # self.logger.log_metrics({'val_epoch_loss': self.epoch_loss['val'][-1]}, epoch=self.current_epoch)
            # for metric_name in self.metrics_fns:
            #     self.logger.log_metrics({'val_epoch_{}'.format(metric_name): self.epoch_metrics['val'][metric_name][-1]}, epoch=self.current_epoch)
            # if self.current_epoch>10: import pdb; pdb.set_trace()
            self.step_outputs['val'].clear()


    def on_test_epoch_end(self) -> None:
        if self.global_rank==0:
            self.save_outputs(process="test")
            self.logger.log_metrics({'test_epoch_loss': self.epoch_loss['test'][-1]}, epoch=self.current_epoch)
            for metric_name in self.metrics_fns:
                self.logger.log_metrics({'test_epoch_{}'.format(metric_name): self.epoch_metrics['test'][metric_name][-1]}, step=self.current_epoch)
            self.step_outputs['test'].clear() 


class TimeShiftProcessor:
    """ Class to process the time shifted data """
    def __init__(self, do_shift: bool, interpolation: bool) -> None:
        self.shift = int(do_shift)
        self.interpolation = interpolation
    
    def shift_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """ Shift and difference the labels """
        # Shift the tensor by the shift duration. If shift duration is positive then shift to the right, else to the left. 
        # If Shift is 0, then return the original tensor
        if self.shift==0: return labels
        zeros_tensor = torch.zeros(labels.shape[0], self.shift, labels.shape[2], device=labels.device)
        shifted_labels = torch.cat([zeros_tensor, labels[:, :-self.shift]], dim=1) if self.shift>0 else torch.cat([labels[:, self.shift:], zeros_tensor], dim=1)
        diffed_labels = labels - shifted_labels
        return diffed_labels

    def reshift_labels(self, output: torch.Tensor, labels: torch.Tensor, mask:torch.Tensor=None) -> torch.Tensor:
        """
            Reshift the output to the original scale. 
            If shift is 0, then return the original output tensor
            Labels are the original labels before shifting
        """
        if self.interpolation: return self.reshift_interpolated_labels(output, labels, mask)
        if self.shift==0: return output
        zeros_tensor = torch.zeros(labels.shape[0], self.shift, labels.shape[2], device=labels.device)
        shifted_labels = torch.cat([zeros_tensor, labels[:, :-self.shift]], dim=1) if self.shift>0 else torch.cat([labels[:, self.shift:], zeros_tensor], dim=1)
        reshifted_output = output + shifted_labels
        # import pdb; pdb.set_trace()
        return reshifted_output
    
    def reshift_interpolated_labels(self, output: torch.Tensor, labels: torch.Tensor, mask:torch.Tensor) -> torch.Tensor:
        """
            Reshift the output to the original scale.
            mask is the outcomes mask.
        """
        if self.shift==0: return output
        zero_tensor = torch.zeros(labels.shape[0], self.shift, labels.shape[2], device=labels.device)
        shifted_labels = torch.cat([zero_tensor, labels[:, :-self.shift]], dim=1) if self.shift>0 else torch.cat([labels[:, self.shift:], zero_tensor], dim=1)
        reshifted_output = output
        
        for seq_id in range(output.shape[0]):
            prev_timestep_present = torch.ones(labels.shape[2], device=labels.device)
            for timestep in range(labels.shape[1]):
                for outcome_idx in range(labels.shape[2]):
                    reshifted_output[seq_id, timestep, outcome_idx] += reshifted_output[seq_id, timestep-1, outcome_idx] if prev_timestep_present[outcome_idx]==0 else shifted_labels[seq_id, timestep, outcome_idx]
                    try:
                        prev_timestep_present[outcome_idx] = mask[seq_id, timestep, outcome_idx]
                    except:
                        import pdb; pdb.set_trace()
        
        return reshifted_output
    
    
class SequenceLengthScheduler:
    """ Class to schedule the sequence length based on the epoch """
    def __init__(self, min_seq_len: int, max_seq_len: int, num_epochs: int, scheduler_type: str='linear') -> None:
        """
            Args:
                min_seq_len (int): Minimum sequence length
                max_seq_len (int): Maximum sequence length
                num_epochs (int): Number of epochs to train for
                scheduler_type (str): Type of scheduler to use. Options: 'linear', 'exponential', 'none'
        """
        assert min_seq_len > 0, "Minimum sequence length must be greater than 0. Minimum sequence length: {}".format(min_seq_len)
        assert max_seq_len > min_seq_len, "Maximum sequence length must be greater than minimum sequence length. Min/Max sequence length: {}/{}".format(min_seq_len, max_seq_len)
        assert num_epochs > 0, "Number of epochs must be greater than 0. Number of epochs: {}".format(num_epochs)
        
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.num_epochs = num_epochs
        self.scheduler_type = scheduler_type
        
    def get_sequence_length(self, current_epoch:int) -> int:
        """
            Get the sequence length based on the current epoch
            Returns:
                int: Sequence length
        """
        if self.scheduler_type == 'linear':
            seq_len = int(self.min_seq_len + (self.max_seq_len - self.min_seq_len) * (current_epoch / self.num_epochs))
        elif self.scheduler_type == 'exponential':
            seq_len = int(self.min_seq_len * ((self.max_seq_len / self.min_seq_len) ** (current_epoch / self.num_epochs)))
        else:
            seq_len = self.max_seq_len  # Default to max_seq_len if unknown scheduler type
            
        seq_len = max(self.min_seq_len, min(seq_len, self.max_seq_len))
        return seq_len
    
    def trim_sequence(self, outcomes_mask: torch.Tensor, current_epoch: int) -> torch.Tensor:
        """
            Sets outcomes_mask to 0 for timesteps beyond the scheduled sequence length. 
            outcomes_mask is of the shape (batch_size, seq_len, num_outcomes). Multiplies the outcomes_mask past get_sequence_length by 0. 
        """
        
        seq_len = self.get_sequence_length(current_epoch)
        if seq_len >= outcomes_mask.shape[1]: return outcomes_mask
        outcomes_mask[:, seq_len:, :] = 0
        return outcomes_mask