from typing import Any, List, Union
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
# import torchmetrics.functional.regression as tm_reg
# import torchmetrics.functional.classification as tm_cls  


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
    
    
def mi_mse(input:torch.Tensor, target:torch.Tensor, reduction:str="within-seq", mask:torch.Tensor=None):
    """
        Calculate MSE loss for Multi Instance Learning. 
        Computes squared loss between input and target for the valid timesteps denoted by the mask. 
    """
    assert reduction in ["within-seq", "flatten", "none", "between-seq"], f"Invalid reduction: {reduction}. Choose from ['within-seq', 'flatten', 'none', 'between-seq']"
    if mask is None:
        mask = torch.ones(input.shape, device=input.device)
    
    if reduction == "within-seq":
        # if isinstance(input, tuple):
        #     # residue = (target - input[0]) # TODO: Just regress the residual as loss
        #     main_loss = torch.square(0.9*target - input[0])*mask
        #     residual_loss = torch.square(0.1*target - input[1])*mask
        #     main_loss = torch.sum(main_loss, axis=1)/torch.sum(mask, axis=1)
        #     main_loss = torch.mean(main_loss, axis=0).mean()
        #     residual_loss = torch.sum(residual_loss, axis=1)/torch.sum(mask, axis=1)
        #     residual_loss = torch.mean(residual_loss, axis=0).mean()
        #     loss = main_loss + residual_loss
        # else:
            loss = torch.square(input - target)*mask
            loss = torch.sum(loss, axis=1)/torch.sum(mask, axis=1)
            loss = torch.mean(loss, axis=0).mean()
    elif reduction == "flatten":
        # if isinstance(input, tuple):
        #     residue = (target - input[0])
        #     main_loss = torch.square(residue)*mask
        #     residual_loss = torch.square(residue - input[1])*mask
        #     main_loss = torch.sum(main_loss, axis=[0, 1])/torch.sum(mask, axis=[0, 1])
        #     residual_loss = torch.sum(residual_loss, axis=[0, 1])/torch.sum(mask, axis=[0, 1])
        #     loss = torch.mean(main_loss) + torch.mean(residual_loss)
        # else:
            loss = torch.square(input - target)*mask
            loss = torch.sum(loss, axis=[0, 1])/torch.sum(mask, axis=[0, 1])
            loss = torch.mean(loss)
    elif reduction == "between-seq":
        # if isinstance(input, tuple):
        #     residue = (target - input[0])
        #     main_loss = torch.square(residue)*mask
        #     residual_loss = torch.square(residue - input[1])*mask
        #     main_loss = torch.sum(main_loss, axis=1)/torch.sum(mask, axis=1)
        #     main_loss = torch.mean(main_loss, axis=0).mean()
        #     residual_loss = torch.sum(residual_loss, axis=1)/torch.sum(mask, axis=1)
        #     residual_loss = torch.mean(residual_loss, axis=0).mean()
        #     loss = main_loss + residual_loss
        # else:
            input_mean = torch.sum(input*mask, axis=1)/torch.sum(mask, axis=1) # average over timesteps
            target_mean = torch.sum(target*mask, axis=1)/torch.sum(mask, axis=1) # average over timesteps
            loss = torch.square(input_mean - target_mean)
            loss = torch.mean(loss, axis=0).mean() # average over sequences and then over outcomes
    elif reduction == "none" or reduction is None:
        loss = torch.square(input - target)*mask
    
    return loss


def mi_smape(input:torch.Tensor, target:torch.Tensor, reduction:str="within-seq", mask:torch.Tensor=None):
    """
        Calculate SMAPE loss for Multi Instance Learning. 
        Computes SMAPE loss between input and target for the valid timesteps denoted by the mask. 
    """
    assert reduction in ["within-seq", "flatten", "none", "between-seq"], f"Invalid reduction: {reduction}. Choose from ['within-seq', 'flatten', 'none', 'between-seq']"
    if mask is None:
        mask = torch.ones(input.shape, device=input.device) 
    
    epsilon = 1e-8
    
    if reduction == "within-seq":
        loss = torch.abs(input - target)/(torch.abs(input) + torch.abs(target) + epsilon)*mask
        loss = 2*torch.sum(loss, axis=1)/torch.sum(mask, axis=1)
        loss = torch.mean(loss, axis=0).mean()
    elif reduction == "flatten":
        loss = torch.abs(input - target)/(torch.abs(input) + torch.abs(target) + epsilon)*mask
        loss = 2*torch.sum(loss, axis=[0, 1])/torch.sum(mask, axis=[0, 1]) # average loss over sequences and timesteps
        loss = torch.mean(loss) # average loss over outcomes 
    elif reduction == "between-seq":
        input_mean = torch.sum(input*mask, axis=1)/torch.sum(mask, axis=1)
        target_mean = torch.sum(target*mask, axis=1)/torch.sum(mask, axis=1)
        loss = torch.abs(input_mean - target_mean)/(torch.abs(input_mean) + torch.abs(target_mean) + epsilon)
        loss = 2*torch.mean(loss, axis=0).mean() # average loss over sequences and then over outcomes
    elif reduction == "none" or reduction is None:
        loss = 2*torch.abs(input - target)/(torch.abs(input) + torch.abs(target))*mask
        
    return loss


def pearson_corrcoef(input, target, mask=None, dim:int=1):
    """
    Calculate Pearson correlation coefficient between input and target.
    Computes Pearson correlation coefficient for the valid timesteps denoted by the mask.
    dim is the dimension along which to compute the correlation.
    Default is 1, which means the correlation is computed along the sequence dimension.
    If mask is None, it assumes all timesteps are valid.
    The input and target should be of the same shape.
    Returns a tensor of Pearson correlation coefficients.
    """
    if mask is None: mask = torch.ones(input.shape, device=input.device)
    input_mean = torch.sum(input*mask, dim=dim)/torch.sum(mask, dim=dim)
    target_mean = torch.sum(target*mask, dim=dim)/torch.sum(mask, dim=dim)
    nr = torch.sum((input - input_mean.unsqueeze(-1))*(target - target_mean.unsqueeze(-1))*mask, dim=dim)
    dr = torch.sqrt(torch.sum((input - input_mean.unsqueeze(-1))**2*mask, dim=dim)*torch.sum((target - target_mean.unsqueeze(-1))**2*mask, dim=dim))
    dr = torch.max(dr, torch.tensor(1e-8, device=input.device))
    return nr/dr


def mi_pearsonr(input:torch.Tensor, target:torch.Tensor, reduction="within-seq", mask:torch.Tensor=None):
    """
        Calculate Pearson correlation coefficient for Multi Instance Learning.
        Computes Pearson correlation coefficient between input and target for the valid timesteps denoted by the mask.
    """
    assert reduction in ["within-seq", "flatten", "between-seq"], f"Invalid reduction: {reduction}. Choose from ['within-seq', 'flatten', 'between-seq']"
    if mask is None:
        mask = torch.ones(input.shape, device=input.device)

    num_outcomes = input.shape[-1]
    pearson_rs = []
    for i in range(num_outcomes):
        if reduction == "flatten":
            # Compute Pearson correlation coefficient for each outcome
            input_ = (input[:, :, i][mask[:, :, i]==1])
            target_ = (target[:, :, i][mask[:, :, i]==1])
            seq_pearsonr = pearson_corrcoef(input_, target_, dim=0) # returns Pearson correlation value on the flattened input and target
            pearson_rs.append(torch.mean(seq_pearsonr)) # Average the Pearson correlation coefficient for all sequences
        elif reduction == "within-seq":
            # Compute Pearson correlation coefficient for each outcome but individually on each sequence
            input_ = input[:, :, i]
            target_ = target[:, :, i]
            mask_ = mask[:, :, i]
            seq_pearsonr = pearson_corrcoef(input_, target_, mask_, dim=1) # returns Pearson correlation value for each sequence
            pearson_rs.append(torch.mean(seq_pearsonr)) # Append the Pearson correlation coefficient for each outcome
        elif reduction == "between-seq":
            # Compute Pearson correlation coefficient for each outcome but on the average of all sequences
            input_mean = torch.sum(input[:, :, i]*mask[:, :, i], dim=1)/torch.sum(mask[:, :, i], dim=1)
            target_mean = torch.sum(target[:, :, i]*mask[:, :, i], dim=1)/torch.sum(mask[:, :, i], dim=1)
            seq_pearsonr = pearson_corrcoef(input_mean, target_mean, dim=0) # returns Pearson correlation value on the average input and target
            pearson_rs.append(seq_pearsonr) 
    # Return the average Pearson correlation coefficient 
    return torch.mean(torch.tensor(pearson_rs))


def mi_mae(input:torch.Tensor, target:torch.Tensor, reduction:str="within-seq", mask:torch.Tensor=None):
    """
        Calculate MAE loss for Multi Instance Learning. 
        Computes MAE loss between input and target for the valid timesteps denoted by the mask. 
    """
    assert reduction in ["within-seq", "flatten", "none", "between-seq"], f"Invalid reduction: {reduction}. Choose from ['within-seq', 'flatten', 'none', 'between-seq']"
    if mask is None:
        mask = torch.ones(input.shape, device=input.device) 
    
    if reduction == "within-seq":
        loss = torch.sum(torch.abs(input - target)*mask, axis=1)/torch.sum(mask, axis=1)
        loss = torch.mean(loss, axis=0).mean()
        # print ("MAE going through within-seq")
        # import pdb; pdb.set_trace()
    elif reduction == "flatten":
        loss = torch.abs(input - target)*mask
        loss = torch.sum(loss, axis=[0, 1])/torch.sum(mask, axis=[0, 1])
        loss = torch.mean(loss)
    elif reduction == "between-seq":
        input_mean = torch.sum(input*mask, axis=1)/torch.sum(mask, axis=1)
        target_mean = torch.sum(target*mask, axis=1)/torch.sum(mask, axis=1)
        loss = torch.abs(input_mean - target_mean)
        loss = torch.mean(loss, axis=0).mean() # average loss over sequences and then over outcomes
    elif reduction == "none" or reduction is None:
        loss = torch.abs(input - target)*mask
    
    return loss

def mi_cs(input:torch.Tensor, target:torch.Tensor, reduction:str="within-seq", mask:torch.Tensor=None):
    """
        Calculate the Cosine Similarity Loss
        Computes the CS loss between input and target for the valid timesteps denoted by the mask
    """
    assert reduction in ["within-seq", "flatten", "between-seq"], f"Invalid reduction: {reduction}. Choose from ['within-seq', 'flatten', 'between-seq']"
    if mask is None:
        mask = torch.ones(input.shape, device=input.device)
    
    if reduction == "within-seq":
        mean_centered_input = input - input.mean(axis=1).unsqueeze(1)
        mean_centered_target = target - target.mean(axis=1).unsqueeze(1)
        nr = torch.sum(mean_centered_input * mean_centered_target * mask, axis=1)
        dr = torch.clamp(torch.norm(mean_centered_input*mask, dim=1)*torch.norm(mean_centered_target*mask, dim=1), min=1e-8)
        cosine_sim = (nr/dr).mean(0).mean()
        loss = 1 - cosine_sim
    elif reduction == "flatten":
        mean_centered_input = input - input.mean()
        mean_centered_target = target - target.mean() 
        nr = torch.sum(mean_centered_input * mean_centered_target * mask, axis=[0, 1])
        dr = torch.clamp(torch.norm(mean_centered_input*mask, dim=[0, 1])*torch.norm(mean_centered_target*mask, dim=[0, 1]), min=1e-8)
        cosine_sim = (nr/dr).mean()
        loss = 1 - cosine_sim
    elif reduction == "between-seq":
        input_mean = torch.sum(input*mask, axis=1)/torch.sum(mask, axis=1)
        target_mean = torch.sum(target*mask, axis=1)/torch.sum(mask, axis=1)
        mean_centered_input_mean = input_mean - input_mean.mean(0).unsqueeze(0)
        mean_centered_target_mean = target_mean - target_mean.mean(0).unsqueeze(0)
        nr = torch.sum(mean_centered_input_mean*mean_centered_target_mean, axis=0)
        dr = torch.clamp(torch.norm(mean_centered_input_mean, dim=0)*torch.norm(mean_centered_target_mean, dim=0), min=1e-8)
        cosine_sim = (nr/dr).mean()
        loss = 1 - cosine_sim
        
    return loss

def mi_nce(
    input: torch.Tensor,          # (B, T, D)
    target: torch.Tensor,         # (B, T, D)
    reduction: str = "flatten",   # "flatten", "within-seq", "between-seq"
    mask: torch.Tensor = None,    # (B,T) or (B,T,D) with 0/1
    tau: float = 0.1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    InfoNCE / NT-Xent computed explicitly with log-sum-exp.
    Diagonal terms are excluded from the denominator in all modes.
    """
    assert reduction in ["flatten", "within-seq", "between-seq"]
    B, T, D = input.shape
    device = input.device
    dtype  = input.dtype

    # Build (B,T) boolean validity
    if mask is None:
        valid_bt = torch.ones((B, T), dtype=torch.bool, device=device)
    else:
        if mask.dim() == 3:
            valid_bt = (mask.sum(dim=-1) > 0)
        elif mask.dim() == 2:
            valid_bt = mask.bool()
        else:
            raise ValueError("mask must have shape (B,T) or (B,T,D)")

    # Helper to make an off-diagonal mask of size N x N
    def offdiag_mask(n: int) -> torch.Tensor:
        eye = torch.eye(n, dtype=torch.bool, device=device)
        return ~eye  # True for off-diagonal

    if reduction == "flatten":
        # Keep only valid (b,t)
        keep = valid_bt.reshape(-1)             # (B*T,)
        if keep.sum() == 0:
            return torch.zeros((), device=device, dtype=dtype)

        x = F.normalize(input.reshape(-1, D)[keep], dim=-1)            # (N,D)
        y = F.normalize(target.reshape(-1, D)[keep].detach(), dim=-1)  # (N,D)

        logits = (x @ y.t()) / tau                                     # (N,N)
        N = logits.size(0)

        pos_logits = logits.diag()                                      # (N,)
        # Remove diagonal from denominator
        mask_off = offdiag_mask(N)
        logits_off = logits.masked_fill(~mask_off, float('-inf'))
        denom = torch.logsumexp(logits_off, dim=1)                      # (N,)

        loss = -(pos_logits - denom).mean()
        return loss

    elif reduction == "within-seq":
        # Cosine over time for each feature; negatives are other sequences (same feature)
        # Apply mask over time
        m_bt1 = valid_bt.unsqueeze(-1).to(dtype)            # (B,T,1)
        x = input * m_bt1                                   # (B,T,D)
        y = target.detach() * m_bt1                         # (B,T,D)

        # L2 norm over time per (b,d)
        x_norm = torch.sqrt((x * x).sum(dim=1) + eps)       # (B,D)
        y_norm = torch.sqrt((y * y).sum(dim=1) + eps)       # (B,D)

        x_hat = x / x_norm.unsqueeze(1)                     # (B,T,D)
        y_hat = y / y_norm.unsqueeze(1)                     # (B,T,D)

        # Cosine over time per feature, across sequences:
        # logits_dbb[d,b,b'] = <x_hat[b,:,d], y_hat[b',:,d]>
        logits_dbb = torch.einsum('btd,Btd->dbB', x_hat, y_hat) / tau   # (D,B,B)

        # Keep only sequences that have at least one valid timestep
        valid_seq = valid_bt.any(dim=1)                                   # (B,)
        if valid_seq.sum() == 0:
            return torch.zeros((), device=device, dtype=dtype)
        keep_idx = valid_seq.nonzero(as_tuple=False).squeeze(1)           # (Bv,)

        logits_dbb = logits_dbb[:, keep_idx][:, :, keep_idx]              # (D,Bv,Bv)
        D_, Bv, _ = logits_dbb.shape

        # Positive logits are the diagonal along the (b,b) axes for each d
        pos_logits = logits_dbb.diagonal(dim1=1, dim2=2)                  # (D,Bv)

        # Denominator excludes diagonal
        off_mask = offdiag_mask(Bv).unsqueeze(0).expand(D_, -1, -1)       # (D,Bv,Bv)
        logits_off = logits_dbb.masked_fill(~off_mask, float('-inf'))     # (D,Bv,Bv)
        denom = torch.logsumexp(logits_off, dim=2)                        # (D,Bv)

        # Loss averaged over sequences, then over features D
        loss_d = -(pos_logits - denom).mean(dim=1)                        # (D,)
        return loss_d.mean()

    else:  # "between-seq"
        # Time-average first (masked) -> (B,D)
        m_bt1 = valid_bt.unsqueeze(-1).to(dtype)                    # (B,T,1)
        counts = valid_bt.sum(dim=1).to(dtype).unsqueeze(1)         # (B,1)
        counts = counts + (counts == 0).to(dtype) * eps             # avoid 0

        x_mean = (input * m_bt1).sum(dim=1) / counts                # (B,D)
        y_mean = (target.detach() * m_bt1).sum(dim=1) / counts      # (B,D)

        # L2 normalize across sequences per feature (cosine on scalars per d)
        # qn[b,d] = x_mean[b,d] / sqrt(sum_b x_mean[b,d]^2 + eps)
        x_den = torch.sqrt((x_mean * x_mean).sum(dim=0, keepdim=True) + eps)  # (1,D)
        y_den = torch.sqrt((y_mean * y_mean).sum(dim=0, keepdim=True) + eps)  # (1,D)
        x_hat = x_mean / x_den                                               # (B,D)
        y_hat = y_mean / y_den                                               # (B,D)

        # For each feature d, outer product over sequences -> (D,B,B)
        logits_dbb = torch.einsum('bd,Bd->dbB', x_hat, y_hat) / tau          # (D,B,B)

        # Drop sequences with no valid timesteps
        valid_seq = valid_bt.any(dim=1)                                      # (B,)
        if valid_seq.sum() == 0:
            return torch.zeros((), device=device, dtype=dtype)
        keep_idx = valid_seq.nonzero(as_tuple=False).squeeze(1)              # (Bv,)

        logits_dbb = logits_dbb[:, keep_idx][:, :, keep_idx]                 # (D,Bv,Bv)
        D_, Bv, _ = logits_dbb.shape

        pos_logits = logits_dbb.diagonal(dim1=1, dim2=2)                     # (D,Bv)

        off_mask = offdiag_mask(Bv).unsqueeze(0).expand(D_, -1, -1)          # (D,Bv,Bv)
        logits_off = logits_dbb.masked_fill(~off_mask, float('-inf'))        # (D,Bv,Bv)
        denom = torch.logsumexp(logits_off, dim=2)                            # (D,Bv)

        loss_d = -(pos_logits - denom).mean(dim=1)                            # (D,)
        return loss_d.mean()

def mi_nce2(
    input: torch.Tensor,          # (B, T, D)  queries
    target: torch.Tensor,         # (B, T, D)  keys (pass .detach() if static)
    reduction: str = "flatten",   # "flatten", "within-seq", "between-seq"
    mask: torch.Tensor = None,    # (B,T) or (B,T,D)
    tau: float = 0.1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Non-negative InfoNCE (includes the positive in the denominator).
    Loss = - ( pos_logit - logaddexp(pos_logit, logsumexp(neg_logits)) )

    Modes
    -----
    flatten:
      Pos: each (b,t,:) vs same (b,t,:) in target
      Neg: all other (b',t',:) in target
      Cosine over D

    within-seq:
      Per-feature D:
        Pos: sequence (b,:,d) vs (b,:,d)
        Neg: all other sequences (b'!=b) for that d
      Cosine over time with mask
      Average loss over D at the end.

    between-seq:
      Time-mean first → (B,D), then per-feature D:
        Pos: mean(b,d) vs mean(b,d)
        Neg: other sequences’ means (b'!=b) for that d
      Average loss over D at the end.
    """
    assert reduction in ["flatten", "within-seq", "between-seq"]
    B, T, D = input.shape
    device, dtype = input.device, input.dtype

    # Build (B,T) validity mask
    if mask is None:
        valid_bt = torch.ones((B, T), dtype=torch.bool, device=device)
    else:
        if mask.dim() == 3:
            valid_bt = (mask.sum(dim=-1) > 0)
        elif mask.dim() == 2:
            valid_bt = mask.bool()
        else:
            raise ValueError("mask must be (B,T) or (B,T,D)")

    def offdiag_mask(n: int) -> torch.Tensor:
        eye = torch.eye(n, dtype=torch.bool, device=device)
        return ~eye

    if reduction == "flatten":
        keep = valid_bt.reshape(-1)  # (B*T,)
        if keep.sum() == 0:
            return torch.zeros((), device=device, dtype=dtype)

        x = F.normalize(input.reshape(-1, D)[keep], dim=-1)            # (N,D)
        y = F.normalize(target.reshape(-1, D)[keep].detach(), dim=-1)  # (N,D)

        logits = (x @ y.t()) / tau                                     # (N,N)
        N = logits.size(0)
        pos_logits = logits.diag()                                      # (N,)

        # negatives exclude diagonal; positive added back via logaddexp
        m_off = offdiag_mask(N)
        logits_off = logits.masked_fill(~m_off, float('-inf'))
        denom = torch.logaddexp(pos_logits, torch.logsumexp(logits_off, dim=1))
        loss = -(pos_logits - denom).mean()
        return loss

    elif reduction == "within-seq":
        # Cosine over time per feature; negatives are other sequences (same feature)
        m_bt1 = valid_bt.unsqueeze(-1).to(dtype)                  # (B,T,1)
        x = input * m_bt1                                         # (B,T,D)
        y = target.detach() * m_bt1                               # (B,T,D)

        # L2 norm over time per (b,d)
        x_den = torch.sqrt((x * x).sum(dim=1) + eps)              # (B,D)
        y_den = torch.sqrt((y * y).sum(dim=1) + eps)              # (B,D)
        x_hat = x / x_den.unsqueeze(1)                            # (B,T,D)
        y_hat = y / y_den.unsqueeze(1)                            # (B,T,D)

        # (D,B,B): for each feature d, all-pairs cosine over time between sequences
        # einsum dims: x_hat[b,t,d] * y_hat[B,t,d] -> (d,b,B)
        logits_dbb = torch.einsum('btd,Btd->dbB', x_hat, y_hat) / tau  # (D,B,B)

        # keep sequences that have at least one valid timestep
        valid_seq = valid_bt.any(dim=1)                                # (B,)
        if valid_seq.sum() == 0:
            return torch.zeros((), device=device, dtype=dtype)
        idx = valid_seq.nonzero(as_tuple=False).squeeze(1)             # (Bv,)
        logits_dbb = logits_dbb[:, idx][:, :, idx]                     # (D,Bv,Bv)
        D_, Bv, _ = logits_dbb.shape

        pos_logits = logits_dbb.diagonal(dim1=1, dim2=2)               # (D,Bv)

        m_off = offdiag_mask(Bv).unsqueeze(0).expand(D_, -1, -1)       # (D,Bv,Bv)
        logits_off = logits_dbb.masked_fill(~m_off, float('-inf'))     # (D,Bv,Bv)
        denom = torch.logaddexp(pos_logits, torch.logsumexp(logits_off, dim=2))  # (D,Bv)

        # avg over sequences, then over features (D)
        loss_d = -(pos_logits - denom).mean(dim=1)                     # (D,)
        return loss_d.mean()

    else:  # "between-seq"
        # Time-average first (masked) -> (B,D)
        m_bt1 = valid_bt.unsqueeze(-1).to(dtype)                       # (B,T,1)
        counts = valid_bt.sum(dim=1).to(dtype).unsqueeze(1)            # (B,1)
        counts = counts + (counts == 0).to(dtype) * eps                # avoid 0

        x_mean = (input * m_bt1).sum(dim=1) / counts                   # (B,D)
        y_mean = (target.detach() * m_bt1).sum(dim=1) / counts         # (B,D)

        # Normalize across sequences per feature (cosine on scalars per d)
        x_den = torch.sqrt((x_mean * x_mean).sum(dim=0, keepdim=True) + eps)  # (1,D)
        y_den = torch.sqrt((y_mean * y_mean).sum(dim=0, keepdim=True) + eps)  # (1,D)
        x_hat = x_mean / x_den                                         # (B,D)
        y_hat = y_mean / y_den                                         # (B,D)

        # For each feature d, all-pairs outer product across sequences -> (D,B,B)
        logits_dbb = torch.einsum('bd,Bd->dbB', x_hat, y_hat) / tau    # (D,B,B)

        valid_seq = valid_bt.any(dim=1)
        if valid_seq.sum() == 0:
            return torch.zeros((), device=device, dtype=dtype)
        idx = valid_seq.nonzero(as_tuple=False).squeeze(1)             # (Bv,)
        logits_dbb = logits_dbb[:, idx][:, :, idx]                     # (D,Bv,Bv)
        D_, Bv, _ = logits_dbb.shape

        pos_logits = logits_dbb.diagonal(dim1=1, dim2=2)               # (D,Bv)
        m_off = offdiag_mask(Bv).unsqueeze(0).expand(D_, -1, -1)       # (D,Bv,Bv)
        logits_off = logits_dbb.masked_fill(~m_off, float('-inf'))     # (D,Bv,Bv)
        denom = torch.logaddexp(pos_logits, torch.logsumexp(logits_off, dim=2))  # (D,Bv)

        loss_d = -(pos_logits - denom).mean(dim=1)                     # (D,)
        return loss_d.mean()
    
def mi_nce3(
    input: torch.Tensor,          # (B, T, D)  -> z_a (anchors/queries)
    target: torch.Tensor,         # (B, T, D)  -> z_b (keys)
    reduction: str = "flatten",   # "flatten", "within-seq", "between-seq"
    mask: torch.Tensor = None,    # (B,T) or (B,T,D)
    tau: float = 0.1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Contrastive InfoNCE where each z_a anchor contrasts against BOTH:
      - other z_b (cross-view negatives) AND
      - other z_a (same-view negatives),
    while *excluding* only the trivial self-pair (z_a[i] vs z_a[i]).
    The positive (z_a[i] vs z_b[i]) IS INCLUDED in the denominator.

    Loss = - ( logit_pos - log( exp(logit_pos) + sum_{neg} exp(logit_neg) ) ).

    Modes:
      flatten:
        instance = (b,t, :)  (cosine over D)
        keys = [z_a_flat_except_self, z_b_flat]  (pos is in z_b block)

      within-seq:
        per-feature D, cosine over time T with mask
        instance = sequence (b,:,d)
        keys = [z_a sequences (exclude self), z_b sequences] for same d
        average loss over D at end

      between-seq:
        time-mean first -> (B,D), per-feature D across sequences
        instance = mean(b,d)
        keys = [means from z_a (exclude self), means from z_b]
        average loss over D at end
    """
    assert reduction in ["flatten", "within-seq", "between-seq"]
    B, T, D = input.shape
    device, dtype = input.device, input.dtype

    # Build (B,T) validity mask
    if mask is None:
        valid_bt = torch.ones((B, T), dtype=torch.bool, device=device)
    else:
        if mask.dim() == 3:
            valid_bt = (mask.sum(dim=-1) > 0)
        elif mask.dim() == 2:
            valid_bt = mask.bool()
        else:
            raise ValueError("mask must be (B,T) or (B,T,D)")

    if reduction == "flatten":
        # Keep only valid (b,t) instances
        keep = valid_bt.reshape(-1)  # (B*T,)
        if keep.sum() == 0:
            return torch.zeros((), device=device, dtype=dtype)

        # Anchors: z_a
        qa = F.normalize(input.reshape(-1, D)[keep], dim=-1)          # (N,D)
        # Keys: both z_a (same view) and z_b (other view)
        ka = F.normalize(input.reshape(-1, D)[keep], dim=-1)          # (N,D)
        kb = F.normalize(target.reshape(-1, D)[keep], dim=-1)         # (N,D) (detach outside if static)

        # logits_aa: (N,N), logits_ab: (N,N)
        logits_aa = (qa @ ka.t()) / tau
        logits_ab = (qa @ kb.t()) / tau

        N = logits_aa.size(0)

        # Build keys = [aa | ab] -> (N, 2N)
        logits = torch.cat([logits_aa, logits_ab], dim=1)

        # Positive is in ab block at column N + i
        idx = torch.arange(N, device=device)
        pos_logits = logits[idx, N + idx]  # (N,)

        # Mask out ONLY the trivial self in the aa block (i vs i); keep everything else
        mask_off = torch.ones((N, 2 * N), dtype=torch.bool, device=device)
        mask_off[idx, idx] = False  # remove self in aa block
        logits_off = logits.masked_fill(~mask_off, float('-inf'))

        # Proper InfoNCE denom = logaddexp(pos, logsumexp(negatives))
        denom = torch.logaddexp(pos_logits, torch.logsumexp(logits_off, dim=1))
        loss = -(pos_logits - denom).mean()
        return loss

    elif reduction == "within-seq":
        # Per-feature cosine over time (masked). Build (D,B,2B) logits for each feature.
        m_bt1 = valid_bt.unsqueeze(-1).to(dtype)          # (B,T,1)
        xa = input * m_bt1                                # (B,T,D)
        xb = target * m_bt1                               # (B,T,D)  (detach outside if static)

        # Normalize sequences over time per (b,d)
        xa_den = torch.sqrt((xa * xa).sum(dim=1) + eps)   # (B,D)
        xb_den = torch.sqrt((xb * xb).sum(dim=1) + eps)   # (B,D)
        xa_hat = xa / xa_den.unsqueeze(1)                 # (B,T,D)
        xb_hat = xb / xb_den.unsqueeze(1)                 # (B,T,D)

        # Keep sequences that have at least one valid timestep
        valid_seq = valid_bt.any(dim=1)                   # (B,)
        if valid_seq.sum() == 0:
            return torch.zeros((), device=device, dtype=dtype)
        keep_idx = valid_seq.nonzero(as_tuple=False).squeeze(1)  # (Bv,)
        Bv = keep_idx.numel()

        xa_hat = xa_hat[keep_idx]                         # (Bv,T,D)
        xb_hat = xb_hat[keep_idx]                         # (Bv,T,D)

        # logits per feature: s_aa = <xa[b,:,d], xa[B,:,d]>, s_ab = <xa[b,:,d], xb[B,:,d]>
        # Shape (D,Bv,Bv) each; then concat along last dim -> (D,Bv,2Bv)
        s_aa = torch.einsum('btd,Btd->dbB', xa_hat, xa_hat) / tau   # (D,Bv,Bv)
        s_ab = torch.einsum('btd,Btd->dbB', xa_hat, xb_hat) / tau   # (D,Bv,Bv)
        logits = torch.cat([s_aa, s_ab], dim=2)                     # (D,Bv,2Bv)

        # Positive is in the ab block at column Bv+i
        ar = torch.arange(Bv, device=device)
        pos_logits = logits[:, ar, Bv + ar]                         # (D,Bv)

        # Mask out only self in aa block
        mask_off = torch.ones((Bv, 2 * Bv), dtype=torch.bool, device=device)
        mask_off[ar, ar] = False
        mask_off = mask_off.unsqueeze(0).expand(D, -1, -1)          # (D,Bv,2Bv)
        logits_off = logits.masked_fill(~mask_off, float('-inf'))

        denom = torch.logaddexp(pos_logits, torch.logsumexp(logits_off, dim=2))  # (D,Bv)
        loss_d = -(pos_logits - denom).mean(dim=1)                   # (D,)
        return loss_d.mean()

    else:  # "between-seq"
        # Time-mean first -> (B,D)
        m_bt1 = valid_bt.unsqueeze(-1).to(dtype)                    # (B,T,1)
        counts = valid_bt.sum(dim=1).to(dtype).unsqueeze(1)         # (B,1)
        counts = counts + (counts == 0).to(dtype) * eps

        xa_mean = (input * m_bt1).sum(dim=1) / counts               # (B,D)
        xb_mean = (target * m_bt1).sum(dim=1) / counts              # (B,D)

        # Normalize across sequences per feature
        xa_den = torch.sqrt((xa_mean * xa_mean).sum(dim=0, keepdim=True) + eps)  # (1,D)
        xb_den = torch.sqrt((xb_mean * xb_mean).sum(dim=0, keepdim=True) + eps)  # (1,D)
        xa_hat = xa_mean / xa_den                                   # (B,D)
        xb_hat = xb_mean / xb_den                                   # (B,D)

        # Keep sequences with any valid timestep
        valid_seq = valid_bt.any(dim=1)
        if valid_seq.sum() == 0:
            return torch.zeros((), device=device, dtype=dtype)
        keep_idx = valid_seq.nonzero(as_tuple=False).squeeze(1)
        xa_hat = xa_hat[keep_idx]                                   # (Bv,D)
        xb_hat = xb_hat[keep_idx]                                   # (Bv,D)
        Bv = xa_hat.size(0)

        # Per-feature logits across sequences
        s_aa = torch.einsum('bd,Bd->dbB', xa_hat, xa_hat) / tau     # (D,Bv,Bv)
        s_ab = torch.einsum('bd,Bd->dbB', xa_hat, xb_hat) / tau     # (D,Bv,Bv)
        logits = torch.cat([s_aa, s_ab], dim=2)                     # (D,Bv,2Bv)

        ar = torch.arange(Bv, device=device)
        pos_logits = logits[:, ar, Bv + ar]                         # (D,Bv)

        mask_off = torch.ones((Bv, 2 * Bv), dtype=torch.bool, device=device)
        mask_off[ar, ar] = False
        mask_off = mask_off.unsqueeze(0).expand(D, -1, -1)
        logits_off = logits.masked_fill(~mask_off, float('-inf'))

        denom = torch.logaddexp(pos_logits, torch.logsumexp(logits_off, dim=2))  # (D,Bv)
        loss_d = -(pos_logits - denom).mean(dim=1)
        return loss_d.mean()