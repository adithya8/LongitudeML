from typing import Any, List, Optional, Union
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
            Calculate regression metrics
        """
        seq_num = kwargs.pop('seq_num', None)
        if seq_num is None:
            # create a seq_num tensor range(len(target))*batch_size
            pass
        
        
    
    
    def classification_metrics(self, preds:torch.Tensor, target:torch.Tensor, **kwargs) -> Any:
        pass
    