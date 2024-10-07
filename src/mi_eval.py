from typing import Any, List, Union
import torch
import torchmetrics.functional.regression as tm_reg
import torchmetrics.functional.classification as tm_cls  


#TODO: Calculate metrics for each timestep range separately (based on how many left to completion)
class MI_Eval:
    def __init__(self, task_type:str, metrics:Union[List[str], str]=None, num_classes:int=1):
        self.task_type = task_type
        self.metrics = metrics
        self.num_classes = num_classes
    
    
    def __call__(self, preds:torch.Tensor, target:torch.Tensor, **kwargs) -> Any:
        if self.task_type == 'regression':
            return self.regression_metrics(preds, target, **kwargs)
        elif self.task_type == 'classification':
            return self.classification_metrics(preds, target, **kwargs)
        else:
            raise Warning(f"Invalid task_type: {self.task_type}")
    
    
    def regression_metrics(self, preds:torch.Tensor, target:torch.Tensor, **kwargs) -> Any:
        """
            Calculate regression metrics (MSE, SMAPE, Pearson) for three cases: 
                1. Last time step only
                2. Half of the time steps
                3. First time step only
        """
        time_ids = kwargs.pop('time_ids', None)
        if time_ids is None:
            # create a time_ids tensor range(len(target))*batch_size
            pass
        
        
    def classification_metrics(self, preds:torch.Tensor, target:torch.Tensor, **kwargs) -> Any:
        """
            Calculate classification metrics (Accuracy, F1, Precision, Recall) for three cases:
                1. Last time step only
                2. Half of the time steps
                3. First time step only
        """
        pass
    
    
def mi_mse(input:torch.Tensor, target:torch.Tensor, reduction:str="mean", mask:torch.Tensor=None):
    """
        Calculate MSE loss for Multi Instance Learning. 
        Computes squared loss between input and target for the valid timesteps denoted by the mask. 
    """
    if mask is None:
        mask = torch.ones(input.shape, device=input.device) 
    
    # TODO Separate the reduction for each outcome. 
    if reduction == "mean":
        loss = torch.sum(torch.square(input - target)*mask, axis=1)/torch.sum(mask, axis=1)
        loss = torch.mean(loss, axis=0).mean()
    elif reduction == "sum":
        loss = torch.sum(torch.square(input - target)*mask, axis=1)
        loss = torch.mean(loss, axis=0).mean()
    elif reduction == "none" or reduction is None:
        loss = torch.square(input - target)*mask
    
    return loss


def mi_smape(input:torch.Tensor, target:torch.Tensor, reduction:str="mean", mask:torch.Tensor=None):
    """
        Calculate SMAPE loss for Multi Instance Learning. 
        Computes SMAPE loss between input and target for the valid timesteps denoted by the mask. 
    """
    if mask is None:
        mask = torch.ones(input.shape, device=input.device) 
    
    epsilon = 1e-8
    
    if reduction == "mean":
        loss = torch.sum(torch.abs(input - target)/(torch.abs(input) + torch.abs(target) + epsilon)*mask, axis=1)/torch.sum(mask, axis=1)
        loss = torch.mean(loss, axis=0).mean()
    elif reduction == "sum":
        loss = torch.sum(torch.abs(input - target)/(torch.abs(input) + torch.abs(target) + epsilon)*mask, axis=1).sum(axis=0)
        loss = torch.mean(loss, axis=0).mean()
    elif reduction == "none" or reduction is None:
        loss = torch.abs(input - target)/(torch.abs(input) + torch.abs(target))*mask
        
    return loss


def mi_pearsonr(input:torch.Tensor, target:torch.Tensor, mask:torch.Tensor=None):
    """
        Calculate Pearson correlation coefficient for Multi Instance Learning.
        Computes Pearson correlation coefficient between input and target for the valid timesteps denoted by the mask.
    """
    if mask is None:
        mask = torch.ones(input.shape, device=input.device)

    num_outcomes = input.shape[-1]
    pearson_rs = []
    for i in range(num_outcomes):
        input_ = (input[:, :, i][mask[:, :, i]==1]).flatten()
        target_ = (target[:, :, i][mask[:, :, i]==1]).flatten()
        axis_pearsonr = tm_reg.pearson_corrcoef(input_, target_)
        pearson_rs.append(axis_pearsonr)
    
    return torch.mean(torch.tensor(pearson_rs))


def mi_mae(input:torch.Tensor, target:torch.Tensor, reduction:str="mean", mask:torch.Tensor=None):
    """
        Calculate MAE loss for Multi Instance Learning. 
        Computes MAE loss between input and target for the valid timesteps denoted by the mask. 
    """
    if mask is None:
        mask = torch.ones(input.shape, device=input.device) 
    
    if reduction == "mean":
        loss = torch.sum(torch.abs(input - target)*mask)/torch.sum(mask)
    elif reduction == "sum":
        loss = torch.sum(torch.abs(input - target)*mask)
    elif reduction == "none" or reduction is None:
        loss = torch.abs(input - target)*mask
    
    return loss