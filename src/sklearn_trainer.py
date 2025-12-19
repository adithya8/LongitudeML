from typing import Any, Dict, Optional, Tuple, List
from copy import deepcopy
import numpy as np
import torch
import pickle
import os
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from .mi_eval import mi_mse, mi_smape, mi_pearsonr, mi_mae


def collect_batch_dict(dataloader) -> Dict[str, torch.Tensor]:
    """
    Collect and concatenate all batches from a dataloader into a single
    batch dictionary.

    This mirrors the behavior of the Lightning path where the model
    receives a rich batch dict with multiple keys (e.g., 'embeddings_*',
    'outcomes', 'outcomes_mask', 'oots_mask', 'ooss_mask', 'time_ids',
    'seq_id', etc.). Here we simply stack all batches along the batch
    dimension for each tensor-valued key.

    Args:
        dataloader: PyTorch DataLoader yielding dict batches

    Returns:
        Dict[str, torch.Tensor]: keys are the same as in individual
        batches; values are concatenated over all batches along dim=0.
    """
    aggregated: Dict[str, List[torch.Tensor]] = {}

    for batch in dataloader:
        for key, value in batch.items():
            # Only aggregate tensor values; ignore others
            if not isinstance(value, torch.Tensor):
                continue
            if key not in aggregated:
                aggregated[key] = [value]
            else:
                aggregated[key].append(value)

    batch_dict: Dict[str, torch.Tensor] = {}
    for key, tensors in aggregated.items():
        batch_dict[key] = torch.cat(tensors, dim=0)

    return batch_dict


def extract_data_from_dataloader(dataloader):
    """
    Legacy helper retained for backward compatibility.

    NOTE: New code should prefer `collect_batch_dict` and let the model's
    `select_features` method decide how to construct X, y, and mask
    from the full batch dictionary.
    """
    batch_dict = collect_batch_dict(dataloader)

    X = batch_dict.get('embeddings', None)
    y = batch_dict.get('outcomes', None)
    mask = batch_dict.get('outcomes_mask', None)
    seq_ids = batch_dict.get('seq_id', batch_dict.get('seq_idx', None))
    time_ids = batch_dict.get('time_ids', None)
    oots_mask = batch_dict.get('oots_mask', None)
    ooss_mask = batch_dict.get('ooss_mask', None)

    return X, y, mask, seq_ids, time_ids, oots_mask, ooss_mask


def reshape_for_sklearn(X, y, mask):
    """
    Reshape 3D tensors to 2D for sklearn models.
    Only keeps valid samples (where mask is True).
    
    **IMPORTANT NOTE ON PARTIAL VALIDITY**:
    This function uses mask.any(dim=-1) which includes timesteps where AT LEAST ONE 
    outcome is valid. For multi-outcome models, this means:
    - If outcomes_mask = [1, 1, 0], this sample IS included in training
    - Sklearn will train on all outcomes, including the invalid one (with mask=0)
    - This is SUBOPTIMAL for multi-outcome sklearn models as sklearn wastes effort
      predicting invalid outcomes
    
    **For single-outcome models**: This is not an issue since there's only one outcome.
    **For multi-outcome models**: Consider using separate models per outcome or 
    modifying this function to use mask.all(dim=-1) to only include fully valid samples.
    
    Args:
        X: (batch_size, seq_len, input_dim) tensor
        y: (batch_size, seq_len, num_outcomes) tensor
        mask: (batch_size, seq_len, num_outcomes) boolean tensor
        
    Returns:
        X_2d: (n_valid_samples, input_dim) numpy array
        y_2d: (n_valid_samples, num_outcomes) numpy array
        valid_indices: list of (batch_idx, time_idx) tuples for reconstruction
    """
    batch_size, seq_len, input_dim = X.shape
    num_outcomes = y.shape[-1]
    
    # Find valid positions (where at least one outcome is valid)
    # NOTE: Uses mask.any(dim=-1) - see docstring for implications on multi-outcome models
    valid_mask = mask.any(dim=-1)  # (batch_size, seq_len)
    
    # Extract valid samples
    X_valid = X[valid_mask]  # (n_valid, input_dim)
    y_valid = y[valid_mask]  # (n_valid, num_outcomes)
    
    # Convert to numpy
    X_2d = X_valid.cpu().numpy()
    y_2d = y_valid.cpu().numpy()
    
    # Store indices for reconstruction
    valid_indices = torch.nonzero(valid_mask, as_tuple=False).tolist()
    
    return X_2d, y_2d, valid_indices


def reconstruct_from_sklearn(predictions, original_shape, valid_indices, mask):
    """
    Reconstruct 3D predictions from sklearn 2D output.
    
    Args:
        predictions: (n_valid_samples, num_outcomes) numpy array
        original_shape: (batch_size, seq_len, num_outcomes) tuple
        valid_indices: list of (batch_idx, time_idx) tuples
        mask: original mask tensor
        
    Returns:
        preds_3d: (batch_size, seq_len, num_outcomes) tensor
    """
    batch_size, seq_len, num_outcomes = original_shape
    
    # Initialize with zeros
    preds_3d = torch.zeros(original_shape)
    
    # Fill in predictions at valid positions
    for idx, (batch_idx, time_idx) in enumerate(valid_indices):
        preds_3d[batch_idx, time_idx, :] = torch.from_numpy(predictions[idx])
    
    return preds_3d


class SklearnModule:
    """
    Wrapper for sklearn models to work with PyTorch dataloaders.
    Similar to PyTorch Lightning's LightningModule but for sklearn models.
    """
    
    def __init__(self, args, model):
        """
        Initialize the sklearn module.
        
        Args:
            args: Namespace or dict with hyperparameters/config
            model: sklearn model instance with fit() and predict() methods
        """
        self.args = args
        self.model = model
        
        # Metrics functions (reuse from mi_eval)
        self.metrics_fns = {
            'mse': mi_mse,
            'smape': mi_smape, 
            'pearsonr': mi_pearsonr,
            'mae': mi_mae
        }
        
        # Storage for predictions and metrics
        self.predictions = dict(train={}, val={}, test={})
        self.metrics = dict(train={}, val={}, test={})

    def _select_features(self, batch_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Ask the model to select which features to use from the aggregated
        batch dictionary.

        The provided `model` is expected to implement:

            select_features(batch_dict: Dict[str, Tensor], args) ->
                (X_3d: Tensor, y_3d: Tensor, mask_3d: Tensor)

        where:
            - X_3d: (batch_size, seq_len, input_dim)
            - y_3d: (batch_size, seq_len, num_outcomes)
            - mask_3d: (batch_size, seq_len, num_outcomes) boolean

        This mirrors the PyTorch design where the model decides which
        embeddings_* keys to use (and how), not the trainer.
        """
        if not hasattr(self.model, "select_features"):
            raise NotImplementedError(
                "The sklearn model passed to SklearnModule must implement "
                "`select_features(batch_dict, args)` and return "
                "(X_3d, y_3d, mask_3d)."
            )

        X_3d, y_3d, mask_3d = self.model.select_features(batch_dict, self.args)

        if not isinstance(X_3d, torch.Tensor) or not isinstance(y_3d, torch.Tensor) or not isinstance(mask_3d, torch.Tensor):
            raise TypeError(
                "select_features must return three torch.Tensor objects: "
                "(X_3d, y_3d, mask_3d)."
            )

        if X_3d.dim() != 3 or y_3d.dim() != 3 or mask_3d.dim() != 3:
            raise ValueError(
                "select_features must return 3D tensors with shapes "
                "(batch_size, seq_len, *). Got shapes: "
                f"X_3d={tuple(X_3d.shape)}, y_3d={tuple(y_3d.shape)}, "
                f"mask_3d={tuple(mask_3d.shape)}"
            )

        return X_3d, y_3d, mask_3d

    def fit(self, dataloader):
        """
        Train the sklearn model on data from dataloader.
        
        Args:
            dataloader: PyTorch DataLoader
            
        Returns:
            Dict with training metrics
        """
        # Aggregate all batches into a single batch dict
        batch_dict = collect_batch_dict(dataloader)

        # Let the model decide which features to use
        X_3d, y_3d, mask_3d = self._select_features(batch_dict)

        # Check for multi-outcome modeling and warn user
        num_outcomes = y_3d.shape[-1]
        if num_outcomes > 1:
            print("\n" + "="*70)
            print("⚠️  WARNING: Multi-outcome modeling with sklearn detected!")
            print(f"   Number of outcomes: {num_outcomes}")
            print("="*70)
            print("Current implementation uses mask.any(dim=-1) which may include")
            print("partially valid samples. This means sklearn will train on some")
            print("invalid outcome values (where outcomes_mask=0).")
            print("\nFor multi-outcome models, consider:")
            print("  1. Training separate sklearn models per outcome, OR")
            print("  2. Modifying reshape_for_sklearn() to use mask.all(dim=-1)")
            print("     to only include fully valid samples")
            print("="*70 + "\n")
        
        # Reshape for sklearn
        X_2d, y_2d, valid_indices = reshape_for_sklearn(X_3d, y_3d, mask_3d)
        
        print(f"Training on {X_2d.shape[0]} valid samples...")
        
        # Fit sklearn model
        self.model.fit(X_2d, y_2d)
        
        # Compute training metrics
        train_preds_2d = self.model.predict(X_2d)
        train_preds_3d = reconstruct_from_sklearn(train_preds_2d, y_3d.shape, valid_indices, mask_3d)

        # Metadata from batch_dict
        seq_ids = batch_dict.get('seq_id', batch_dict.get('seq_idx', None))
        time_ids = batch_dict.get('time_ids', None)
        oots_mask = batch_dict.get('oots_mask', None)
        ooss_mask = batch_dict.get('ooss_mask', None)
        
        # Store predictions
        self.predictions['train'] = {
            'preds': train_preds_3d,
            'outcomes': y_3d,
            'outcomes_mask': mask_3d,
            'seq_id': seq_ids,
            'time_ids': time_ids,
            'oots_mask': oots_mask,
            'ooss_mask': ooss_mask
        }
        
        # Compute metrics
        train_metrics = self.compute_metrics(train_preds_3d, y_3d, mask_3d)
        self.metrics['train'] = train_metrics
        
        return train_metrics
    
    def evaluate(self, dataloader, split='val'):
        """
        Evaluate the sklearn model on data from dataloader.
        Computes metrics for multiple subsets based on oots_mask and ooss_mask:
        - ws_wt: within sample, within time (ooss==0 & oots==0)
        - ws_oots: within sample, out of time (ooss==0 & oots==1)
        - wt_ooss: within time, out of sample (ooss==1 & oots==0)
        - oots_ooss: out of time, out of sample (ooss==1 & oots==1)
        - oots: all out of time samples (oots==1)
        - ooss: all out of sample sequences (ooss==1)
        - valset: all validation data (oots==1 OR ooss==1)
        
        Args:
            dataloader: PyTorch DataLoader
            split: 'val' or 'test'
            
        Returns:
            Dict with evaluation metrics for all subsets
        """
        # Aggregate all data from dataloader
        batch_dict = collect_batch_dict(dataloader)

        # Let the model decide which features to use
        X_3d, y_3d, mask_3d = self._select_features(batch_dict)

        # Reshape for sklearn
        X_2d, y_2d, valid_indices = reshape_for_sklearn(X_3d, y_3d, mask_3d)
        
        print(f"Evaluating on {X_2d.shape[0]} valid samples...")
        
        # Predict
        preds_2d = self.model.predict(X_2d)
        preds_3d = reconstruct_from_sklearn(preds_2d, y_3d.shape, valid_indices, mask_3d)

        seq_ids = batch_dict.get('seq_id', batch_dict.get('seq_idx', None))
        time_ids = batch_dict.get('time_ids', None)
        oots_mask = batch_dict.get('oots_mask', None)
        ooss_mask = batch_dict.get('ooss_mask', None)
        
        # Store predictions
        self.predictions[split] = {
            'preds': preds_3d,
            'outcomes': y_3d,
            'outcomes_mask': mask_3d,
            'seq_id': seq_ids,
            'time_ids': time_ids,
            'oots_mask': oots_mask,
            'ooss_mask': ooss_mask
        }
        
        # Compute metrics for all subsets
        metrics = self.compute_exhaustive_metrics(preds_3d, y_3d, mask_3d, oots_mask, ooss_mask)
        self.metrics[split] = metrics
        
        return metrics
    
    def predict(self, dataloader):
        """
        Make predictions without computing metrics (no labels needed).
        
        Args:
            dataloader: PyTorch DataLoader
            
        Returns:
            Dict with predictions and metadata
        """
        # Aggregate full batch dict
        batch_dict = collect_batch_dict(dataloader)

        # For prediction, we still rely on select_features to decide X
        # If labels are not present, the model can return dummy y/mask or
        # the caller should handle that separately.
        y = batch_dict.get('outcomes', None)
        mask = batch_dict.get('outcomes_mask', None)

        if y is None:
            # No labels provided, predict all timesteps
            # We still need to choose an input tensor; use select_features
            X_3d, _, _ = self._select_features(batch_dict)
            batch_size, seq_len, input_dim = X_3d.shape
            # Flatten all timesteps
            X_2d = X_3d.reshape(-1, input_dim).cpu().numpy()
            valid_indices = [(i, j) for i in range(batch_size) for j in range(seq_len)]
            # Assume single outcome for shape
            num_outcomes = 1  # Will be determined from predictions
        else:
            # Use model + mask to determine valid samples
            X_3d, y, mask = self._select_features(batch_dict)
            X_2d, y_2d, valid_indices = reshape_for_sklearn(X_3d, y, mask)
            num_outcomes = y.shape[-1]
        
        print(f"Predicting on {X_2d.shape[0]} samples...")
        
        # Predict
        preds_2d = self.model.predict(X_2d)
        
        # Determine shape
        if y is not None:
            original_shape = y.shape
        else:
            batch_size, seq_len, _ = X_3d.shape
            if len(preds_2d.shape) == 1:
                num_outcomes = 1
                preds_2d = preds_2d.reshape(-1, 1)
            else:
                num_outcomes = preds_2d.shape[-1]
            original_shape = (batch_size, seq_len, num_outcomes)
        
        preds_3d = reconstruct_from_sklearn(
            preds_2d,
            original_shape,
            valid_indices,
            mask if mask is not None else torch.ones(original_shape),
        )
        
        return {
            'preds': preds_3d,
            'outcomes': y,
            'outcomes_mask': mask,
            'seq_id': batch_dict.get('seq_id', batch_dict.get('seq_idx', None)),
            'time_ids': batch_dict.get('time_ids', None),
            'oots_mask': batch_dict.get('oots_mask', None),
            'ooss_mask': batch_dict.get('ooss_mask', None)
        }
    
    def compute_metrics(self, preds, targets, mask):
        """
        Compute evaluation metrics using mi_eval functions.
        
        Args:
            preds: (batch_size, seq_len, num_outcomes) tensor
            targets: (batch_size, seq_len, num_outcomes) tensor
            mask: (batch_size, seq_len, num_outcomes) tensor
            
        Returns:
            Dict with metric values
        """
        metrics = {}
        reduction = getattr(self.args, 'metrics_reduction', 'within-seq')
        
        for metric_name, metric_fn in self.metrics_fns.items():
            try:
                value = metric_fn(input=preds, target=targets, mask=mask, reduction=reduction)
                metrics[metric_name] = value.item() if torch.is_tensor(value) else value
            except Exception as e:
                print(f"Warning: Could not compute {metric_name}: {e}")
                metrics[metric_name] = None
        
        return metrics
    
    def compute_exhaustive_metrics(self, preds, targets, mask, oots_mask, ooss_mask):
        """
        Compute metrics for multiple data subsets based on oots_mask and ooss_mask.
        Similar to validation_step in MILightningModule.
        
        Args:
            preds: (batch_size, seq_len, num_outcomes) tensor
            targets: (batch_size, seq_len, num_outcomes) tensor
            mask: (batch_size, seq_len, num_outcomes) tensor
            oots_mask: (batch_size, seq_len) tensor - out of time indicator
            ooss_mask: (batch_size,) or (batch_size, 1) tensor - out of sample indicator
            
        Returns:
            Dict with metrics for all subsets
        """
        reduction = self.args.metrics_reduction
        all_metrics = {}
        
        # Default values for missing subsets
        default_value = -1.0
        # Basic shape checks
        if preds is None or targets is None or mask is None:
            raise ValueError("preds, targets, and mask must not be None.")

        if preds.shape != targets.shape or preds.shape != mask.shape:
            raise ValueError(
                f"preds, targets, and mask must have the same shape. "
                f"Got preds={tuple(preds.shape)}, targets={tuple(targets.shape)}, "
                f"mask={tuple(mask.shape)}"
            )

        batch_size, seq_len, _ = preds.shape

        if oots_mask is None or ooss_mask is None:
            raise ValueError("oots_mask and ooss_mask must not be None.")

        # oots_mask should be time-wise: (B, T)
        if oots_mask.dim() == 1:
            # Broadcast a sequence-level oots_mask across timesteps if needed
            oots_time = oots_mask.unsqueeze(-1).expand(-1, seq_len)
        elif oots_mask.dim() == 2:
            oots_time = oots_mask
        else:
            raise ValueError(
                f"oots_mask must have dim 1 or 2. Got shape {tuple(oots_mask.shape)}"
            )

        if oots_time.shape[0] != batch_size or oots_time.shape[1] != seq_len:
            raise ValueError(
                f"oots_mask shape {tuple(oots_time.shape)} is incompatible with "
                f"preds/mask shape {tuple(preds.shape)}"
            )

        # ooss_mask is sequence-wise: (B,) or (B,1)
        if ooss_mask.dim() == 1:
            ooss_seq = ooss_mask
        elif ooss_mask.dim() == 2 and ooss_mask.shape[1] == 1:
            ooss_seq = ooss_mask[:, 0]
        else:
            # Fallback: take the first column as sequence-level indicator
            ooss_seq = ooss_mask[:, 0]

        if ooss_seq.shape[0] != batch_size:
            raise ValueError(
                f"ooss_mask sequence dimension {tuple(ooss_seq.shape)} is incompatible "
                f"with preds/mask shape {tuple(preds.shape)}"
            )

        # Helper to compute metrics for a given subset condition
        def compute_subset_metrics(subset_name: str, cond: torch.Tensor):
            """
            cond: boolean tensor broadcastable to mask shape (B, T, 1 or num_outcomes)
            """
            subset_mask = torch.where(cond, mask, torch.zeros_like(mask))
            valid_seqs = torch.sum(subset_mask, dim=[1, 2]) > 0

            if torch.sum(valid_seqs) == 0:
                for metric_name in self.metrics_fns.keys():
                    all_metrics[f'{subset_name}_{metric_name}'] = default_value
                return

            subset_preds = preds[valid_seqs]
            subset_targets = targets[valid_seqs]
            subset_mask_filtered = subset_mask[valid_seqs]

            for metric_name, metric_fn in self.metrics_fns.items():
                try:
                    value = metric_fn(
                        input=subset_preds,
                        target=subset_targets,
                        mask=subset_mask_filtered,
                        reduction=reduction,
                    )
                    all_metrics[f'{subset_name}_{metric_name}'] = (
                        value.item() if torch.is_tensor(value) else value
                    )
                except Exception:
                    all_metrics[f'{subset_name}_{metric_name}'] = default_value

        # 1. ws_wt: within sample, within time (ooss==0 & oots==0)
        cond_ws_wt = (
            (ooss_seq == 0).unsqueeze(-1).unsqueeze(-1)
            & (oots_time == 0).unsqueeze(-1)
        )
        compute_subset_metrics("ws_wt", cond_ws_wt)

        # 2. valset: all validation data (oots==1 OR ooss==1)
        cond_valset = (
            (ooss_seq == 1).unsqueeze(-1).unsqueeze(-1)
            | (oots_time == 1).unsqueeze(-1)
        )
        compute_subset_metrics("valset", cond_valset)

        # 3. ws_oots: within sample, out of time (ooss==0 & oots==1)
        cond_ws_oots = (
            (ooss_seq == 0).unsqueeze(-1).unsqueeze(-1)
            & (oots_time == 1).unsqueeze(-1)
        )
        compute_subset_metrics("ws_oots", cond_ws_oots)

        # 4. wt_ooss: within time, out of sample (ooss==1 & oots==0)
        cond_wt_ooss = (
            (ooss_seq == 1).unsqueeze(-1).unsqueeze(-1)
            & (oots_time == 0).unsqueeze(-1)
        )
        compute_subset_metrics("wt_ooss", cond_wt_ooss)

        # 5. oots_ooss: out of time, out of sample (ooss==1 & oots==1)
        cond_oots_ooss = (
            (ooss_seq == 1).unsqueeze(-1).unsqueeze(-1)
            & (oots_time == 1).unsqueeze(-1)
        )
        compute_subset_metrics("oots_ooss", cond_oots_ooss)

        # 6. oots: all out of time samples (oots==1)
        cond_oots = (oots_time == 1).unsqueeze(-1)
        compute_subset_metrics("oots", cond_oots)

        # 7. ooss: all out of sample sequences (ooss==1)
        cond_ooss = (ooss_seq == 1).unsqueeze(-1).unsqueeze(-1)
        compute_subset_metrics("ooss", cond_ooss)

        return all_metrics


class SklearnTrainer:
    """
    Trainer class for sklearn models, similar to PyTorch Lightning's Trainer.
    Orchestrates training, validation, testing, and hyperparameter search.
    """
    
    def __init__(self, output_dir, logger=None, **trainer_params):
        """
        Initialize the sklearn trainer.
        
        Args:
            output_dir: Directory to save models and results
            logger: Logger instance (comet, tensorboard, etc.)
            **trainer_params: Additional trainer configuration
        """
        self.output_dir = output_dir
        self.logger = logger
        self.trainer_params = trainer_params
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def fit(self, module, train_dataloader, val_dataloader=None):
        """
        Train the model and optionally validate.
        
        Args:
            module: SklearnModule instance
            train_dataloader: Training data loader
            val_dataloader: Optional validation data loader
            
        Returns:
            Dict with training (and validation) metrics
        """
        print("=" * 60)
        print("Training...")
        print("=" * 60)
        
        # Train
        train_metrics = module.fit(train_dataloader)
        
        # Log training metrics
        if self.logger:
            for metric_name, value in train_metrics.items():
                if value is not None:
                    self.logger.log_metrics({f'train_{metric_name}': value}, step=0)
        
        print(f"Training metrics: {train_metrics}")
        
        # Validate if dataloader provided
        val_metrics = None
        if val_dataloader is not None:
            val_metrics = self.validate(module, val_dataloader)
        
        # Save model
        self.save_model(module, 'final_model.pkl')
        
        return {'train': train_metrics, 'val': val_metrics}
    
    def validate(self, module, val_dataloader):
        """
        Validate the model (can be called anytime).
        
        Args:
            module: SklearnModule instance
            val_dataloader: Validation data loader
            
        Returns:
            Dict with validation metrics
        """
        print("=" * 60)
        print("Validating...")
        print("=" * 60)
        
        val_metrics = module.evaluate(val_dataloader, split='val')
        
        # Log validation metrics (skip default values of -1.0 or -2.0)
        if self.logger:
            for metric_name, value in val_metrics.items():
                if value is not None and value >= -0.5:  # Skip default values
                    self.logger.log_metrics({f'val_{metric_name}': value}, step=0)
        
        print(f"Validation metrics: {val_metrics}")
        
        return val_metrics
    
    def test(self, module, test_dataloader):
        """
        Test the model.
        
        Args:
            module: SklearnModule instance
            test_dataloader: Test data loader
            
        Returns:
            Dict with test metrics
        """
        print("=" * 60)
        print("Testing...")
        print("=" * 60)
        
        test_metrics = module.evaluate(test_dataloader, split='test')
        
        # Log test metrics (skip default values of -1.0 or -2.0)
        if self.logger:
            for metric_name, value in test_metrics.items():
                if value is not None and value >= -0.5:  # Skip default values
                    self.logger.log_metrics({f'test_{metric_name}': value}, step=0)
        
        print(f"Test metrics: {test_metrics}")
        
        return test_metrics
    
    def predict(self, module, predict_dataloader):
        """
        Make predictions without labels.
        
        Args:
            module: SklearnModule instance
            predict_dataloader: Data loader (may not have labels)
            
        Returns:
            Dict with predictions and metadata
        """
        print("=" * 60)
        print("Predicting...")
        print("=" * 60)
        
        predictions = module.predict(predict_dataloader)
        
        return predictions
    
    def hyperparameter_search(self, module, param_grid, train_dataloader, 
                             val_dataloader, search_type='grid', n_iter=10, cv=3):
        """
        Perform hyperparameter search using GridSearchCV or RandomizedSearchCV.
        
        Args:
            module: SklearnModule instance (will be cloned)
            param_grid: Dict of parameter names to lists of values
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            search_type: 'grid' or 'random'
            n_iter: Number of iterations for random search
            cv: Number of cross-validation folds
            
        Returns:
            SklearnModule with best model
        """
        print("=" * 60)
        print(f"Hyperparameter search ({search_type})...")
        print("=" * 60)
        
        # Extract training data
        X_train, y_train, mask_train, _, _, _, _ = extract_data_from_dataloader(train_dataloader)
        X_train_2d, y_train_2d, _ = reshape_for_sklearn(X_train, y_train, mask_train)
        
        # Extract validation data for final evaluation
        X_val, y_val, mask_val, seq_ids_val, time_ids_val, oots_val, ooss_val = extract_data_from_dataloader(val_dataloader)
        X_val_2d, y_val_2d, valid_indices_val = reshape_for_sklearn(X_val, y_val, mask_val)
        
        # Create search object
        if search_type == 'grid':
            search = GridSearchCV(
                module.model, 
                param_grid, 
                cv=cv,
                scoring='neg_mean_squared_error',
                verbose=2,
                n_jobs=-1
            )
        elif search_type == 'random':
            search = RandomizedSearchCV(
                module.model,
                param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring='neg_mean_squared_error',
                verbose=2,
                n_jobs=-1,
                random_state=getattr(module.args, 'seed', 42)
            )
        else:
            raise ValueError(f"Invalid search_type: {search_type}. Use 'grid' or 'random'.")
        
        # Perform search
        print(f"Searching over {len(param_grid)} parameters...")
        search.fit(X_train_2d, y_train_2d)
        
        print(f"Best parameters: {search.best_params_}")
        print(f"Best CV score: {-search.best_score_:.4f}")
        
        # Log best parameters
        if self.logger:
            self.logger.log_hyperparams(search.best_params_)
            self.logger.log_metrics({'cv_best_mse': -search.best_score_}, step=0)
        
        # Create new module with best model
        best_module = SklearnModule(module.args, search.best_estimator_)
        
        # Evaluate on validation set
        val_preds_2d = search.best_estimator_.predict(X_val_2d)
        val_preds_3d = reconstruct_from_sklearn(val_preds_2d, y_val.shape, valid_indices_val, mask_val)
        
        # Store predictions
        best_module.predictions['val'] = {
            'preds': val_preds_3d,
            'outcomes': y_val,
            'outcomes_mask': mask_val,
            'seq_id': seq_ids_val,
            'time_ids': time_ids_val,
            'oots_mask': oots_val,
            'ooss_mask': ooss_val
        }
        
        # Compute validation metrics (exhaustive)
        val_metrics = best_module.compute_exhaustive_metrics(val_preds_3d, y_val, mask_val, oots_val, ooss_val)
        best_module.metrics['val'] = val_metrics
        
        print(f"Validation metrics with best model: {val_metrics}")
        
        # Log validation metrics (skip default values)
        if self.logger:
            for metric_name, value in val_metrics.items():
                if value is not None and value >= -0.5:  # Skip default values
                    self.logger.log_metrics({f'val_best_{metric_name}': value}, step=0)
        
        # Save best model
        self.save_model(best_module, 'best_model.pkl')
        
        return best_module
    
    def _get_save_dir(self):
        """
        Get the directory for saving models/results.
        Uses logger's project_name and experiment_key if available.
        
        Returns:
            Path string
        """
        if self.logger and hasattr(self.logger, '_project_name') and hasattr(self.logger, '_experiment_key'):
            save_dir = os.path.join(self.output_dir, self.logger._project_name, self.logger._experiment_key)
        else:
            save_dir = self.output_dir
        
        os.makedirs(save_dir, exist_ok=True)
        return save_dir
    
    def save_model(self, module, filename):
        """
        Save the module (including the sklearn model) to disk.
        Uses output_dir/project_name/experiment_key/ structure when logger is available.
        
        Args:
            module: SklearnModule instance
            filename: Name of the file to save
        """
        save_dir = self._get_save_dir()
        filepath = os.path.join(save_dir, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': module.model,
                'args': module.args,
                'metrics': module.metrics,
                'predictions': module.predictions
            }, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filename):
        """
        Load a saved module from disk.
        Uses output_dir/project_name/experiment_key/ structure when logger is available.
        
        Args:
            filename: Name of the file to load
            
        Returns:
            SklearnModule instance
        """
        save_dir = self._get_save_dir()
        filepath = os.path.join(save_dir, filename)
        
        with open(filepath, 'rb') as f:
            saved_data = pickle.load(f)
        
        module = SklearnModule(saved_data['args'], saved_data['model'])
        module.metrics = saved_data['metrics']
        module.predictions = saved_data['predictions']
        
        print(f"Model loaded from {filepath}")
        
        return module



