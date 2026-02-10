from typing import Any, Dict, Optional, Tuple, List
from copy import deepcopy
import numpy as np
import torch
import pickle
import os
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, ParameterGrid, ParameterSampler

from .mi_eval import mi_mse, mi_smape, mi_pearsonr, mi_mae
from .mi_lightningmodule import TimeShiftProcessor


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
    
    print(f"[DEBUG] reshape_for_sklearn - Input shapes:")
    print(f"  X shape: {X.shape}, y shape: {y.shape}, mask shape: {mask.shape}")
    
    # Find valid positions (where at least one outcome is valid)
    # NOTE: Uses mask.any(dim=-1) - see docstring for implications on multi-outcome models
    valid_mask = mask.any(dim=-1)  # (batch_size, seq_len)
    n_valid = valid_mask.sum().item()
    print(f"  Valid mask shape: {valid_mask.shape}, valid samples: {n_valid}/{batch_size * seq_len}")
    
    # Extract valid samples
    X_valid = X[valid_mask]  # (n_valid, input_dim)
    y_valid = y[valid_mask]  # (n_valid, num_outcomes)
    
    # Convert to numpy
    X_2d = X_valid.cpu().numpy()
    y_2d = y_valid.cpu().numpy()
    
    print(f"[DEBUG] reshape_for_sklearn - Output shapes:")
    print(f"  X_2d shape: {X_2d.shape}, y_2d shape: {y_2d.shape}")
    
    # Store indices for reconstruction
    valid_indices = torch.nonzero(valid_mask, as_tuple=False).tolist()
    
    return X_2d, y_2d, valid_indices


def reconstruct_from_sklearn(predictions, original_shape, valid_indices, mask):
    """
    Reconstruct 3D predictions from sklearn 2D output.
    
    Args:
        predictions: (n_valid_samples, num_outcomes) numpy array, or (n_valid_samples,) for single outcome
        original_shape: (batch_size, seq_len, num_outcomes) tuple
        valid_indices: list of (batch_idx, time_idx) tuples
        mask: original mask tensor
        
    Returns:
        preds_3d: (batch_size, seq_len, num_outcomes) tensor
    """
    batch_size, seq_len, num_outcomes = original_shape
    
    # Ensure predictions is 2D: (n_valid_samples, num_outcomes)
    predictions = np.asarray(predictions)
    if predictions.ndim == 1:
        # Single outcome case: reshape to (n_valid_samples, 1)
        predictions = predictions.reshape(-1, 1)
    elif predictions.ndim == 0:
        # Scalar case (shouldn't happen, but handle it)
        predictions = np.array([[predictions.item()]])
    
    # Initialize with zeros
    preds_3d = torch.zeros(original_shape)
    
    # Fill in predictions at valid positions
    for idx, (batch_idx, time_idx) in enumerate(valid_indices):
        pred_value = predictions[idx]  # This is now guaranteed to be an array
        if isinstance(pred_value, np.ndarray):
            preds_3d[batch_idx, time_idx, :] = torch.from_numpy(pred_value)
        else:
            # Fallback for scalar (shouldn't happen after reshape, but just in case)
            preds_3d[batch_idx, time_idx, :] = torch.tensor([pred_value])
    
    return preds_3d


def build_sequence_time_cv_masks(
    X_3d: torch.Tensor,
    y_3d: torch.Tensor,
    outcomes_mask_3d: torch.Tensor,
    n_folds: int = 3,
    stratify: bool = True,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Build custom CV masks that respect both sequence-wise and time-wise structure.

    - Cross-sectional: split sequences into n_folds folds.
      - If stratify=True: sequences are sorted by mean outcome (computed only on valid
        timesteps), breaking ties by std dev when means match up to 4 decimal places,
        then assigned to folds using round-robin.
      - If stratify=False: sequences are split into n_folds contiguous blocks.
    - Prospective: global time cutoff applied uniformly to ALL sequences.
      - First 2/3 of sequence length (timesteps < floor(2/3 * seq_len)) are train-time.
      - Last 1/3 of sequence length (timesteps >= floor(2/3 * seq_len)) are eval-time.
      - Only valid timesteps (where outcomes_mask_3d is True) are considered.

    For fold k (holding out sequences S_k):
      - Train positions:
          * sequences in S_train = all_sequences \\ S_k
          * timesteps in train-time (first 2/3 of sequence length, where valid)
      - Eval positions (union):
          * longitudinal eval (fixed across folds):
              sequences in S_train, timesteps in eval-time (last 1/3 of sequence length, where valid)
          * within-time OOSS:
              sequences in S_k, timesteps in train-time (first 2/3 of sequence length, where valid)

    Args:
        X_3d: (batch_size, seq_len, input_dim) tensor
        y_3d: (batch_size, seq_len, num_outcomes) tensor
        outcomes_mask_3d: (batch_size, seq_len, num_outcomes) boolean tensor
        n_folds: number of CV folds (typically 3)
        stratify: if True, stratify sequences by outcome mean/std; if False, use contiguous blocks

    Returns:
        List of length n_folds, each element is a tuple
        (train_mask_2d, eval_mask_2d, train_mask_2d_unfiltered, eval_mask_2d_unfiltered),
        all (batch_size, seq_len) boolean tensors.
        - train_mask_2d, eval_mask_2d: Filtered masks (only valid timesteps)
        - train_mask_2d_unfiltered, eval_mask_2d_unfiltered: Unfiltered masks (all positions in split)
    """
    if X_3d.dim() != 3 or y_3d.dim() != 3 or outcomes_mask_3d.dim() != 3:
        raise ValueError(
            f"build_sequence_time_cv_masks expects 3D tensors. "
            f"Got X_3d={tuple(X_3d.shape)}, y_3d={tuple(y_3d.shape)}, "
            f"outcomes_mask_3d={tuple(outcomes_mask_3d.shape)}"
        )

    batch_size, seq_len, _ = X_3d.shape

    # Determine which timesteps have at least one valid outcome
    # valid_time[s, t] == True means at least one outcome is valid at (s, t)
    valid_time = outcomes_mask_3d.any(dim=-1)  # (B, T)

    # Assign sequences to folds
    if stratify:
        # Compute mean and std per sequence (only on valid timesteps)
        seq_stats = []
        for s in range(batch_size):
            # Get valid timesteps for this sequence
            vt = valid_time[s]  # (T,)
            if not vt.any():
                # No valid outcomes; use dummy stats (will be sorted last)
                seq_stats.append((float('inf'), float('inf'), s))
                continue
            
            # Extract valid outcomes for this sequence
            y_seq = y_3d[s, vt]  # (n_valid, num_outcomes)
            mask_seq = outcomes_mask_3d[s, vt]  # (n_valid, num_outcomes)
            
            # For multi-outcome, compute mean/std across all outcomes at valid positions
            # Flatten to get all valid outcome values
            valid_outcomes = y_seq[mask_seq].cpu().numpy()
            
            if len(valid_outcomes) == 0:
                seq_stats.append((float('inf'), float('inf'), s))
                continue
            
            mean_val = float(np.mean(valid_outcomes))
            std_val = float(np.std(valid_outcomes))
            seq_stats.append((mean_val, std_val, s))
        
        # Sort by mean (ascending), breaking ties by std (ascending)
        # Round means to 4 decimal places for tie-breaking comparison
        seq_stats_sorted = sorted(
            seq_stats,
            key=lambda x: (round(x[0], 4), x[1])
        )
        
        # Round-robin assignment: sorted_index % n_folds
        seq_fold_assignments = np.zeros(batch_size, dtype=np.int64)
        for sorted_idx, (_, _, orig_idx) in enumerate(seq_stats_sorted):
            fold_id = sorted_idx % n_folds
            seq_fold_assignments[orig_idx] = fold_id
        
        # Group sequences by fold
        seq_folds = []
        for k in range(n_folds):
            fold_seqs = np.where(seq_fold_assignments == k)[0]
            seq_folds.append(fold_seqs)
    else:
        # Contiguous blocks (original behavior)
        seq_indices = np.arange(batch_size)
        seq_folds = np.array_split(seq_indices, n_folds)

    # Compute global time cutoff: first 2/3 of sequence length are train-time, last 1/3 are eval-time
    # This is applied uniformly to ALL sequences (not per-sequence)
    time_cutoff = int(np.floor(2.0 * seq_len / 3.0))
    if time_cutoff <= 0:
        time_cutoff = 1  # At least split at position 1

    cv_masks: List[Tuple[torch.Tensor, torch.Tensor]] = []

    for k in range(n_folds):
        held_out_seqs = seq_folds[k]
        held_out_mask = torch.zeros(batch_size, dtype=torch.bool)
        if len(held_out_seqs) > 0:
            held_out_mask[torch.tensor(held_out_seqs, dtype=torch.long)] = True

        # Train/eval masks over (B, T)
        train_mask_2d = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        eval_mask_2d = torch.zeros(batch_size, seq_len, dtype=torch.bool)

        for s in range(batch_size):
            vt = valid_time[s]  # (T,)
            
            if not vt.any():
                # No valid outcomes in this sequence; skip
                continue

            # Global time split: timesteps < time_cutoff are train-time, >= time_cutoff are eval-time
            # But only consider valid timesteps
            train_time_mask = torch.zeros(seq_len, dtype=torch.bool)
            eval_time_mask = torch.zeros(seq_len, dtype=torch.bool)
            train_time_mask[:time_cutoff] = True
            eval_time_mask[time_cutoff:] = True

            # Apply time split only to valid timesteps
            train_positions = train_time_mask & vt
            eval_positions = eval_time_mask & vt

            if not held_out_mask[s]:
                # Sequence is in-sample for this fold
                # Train on its train-time timesteps
                train_mask_2d[s, train_positions] = True
                # Longitudinal eval on its eval-time timesteps
                eval_mask_2d[s, eval_positions] = True
            else:
                # Sequence is out-of-sample for this fold
                # Within-time OOSS: eval on its train-time timesteps
                eval_mask_2d[s, train_positions] = True
                # (Optionally, could also eval on eval_positions for OOTS+OOSS)

        # Save unfiltered masks for statistics (before filtering by valid_time)
        train_mask_2d_unfiltered = train_mask_2d.clone()
        eval_mask_2d_unfiltered = eval_mask_2d.clone()
        
        # Ensure we never select invalid-outcome positions
        train_mask_2d = train_mask_2d & valid_time
        eval_mask_2d = eval_mask_2d & valid_time

        cv_masks.append((train_mask_2d, eval_mask_2d, train_mask_2d_unfiltered, eval_mask_2d_unfiltered))

    return cv_masks


def flatten_with_custom_mask(
    X_3d: torch.Tensor,
    y_3d: torch.Tensor,
    mask_3d: torch.Tensor,
    position_mask_2d: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Flatten 3D tensors to 2D for sklearn, using a custom position mask.

    Args:
        X_3d: (batch_size, seq_len, input_dim) tensor
        y_3d: (batch_size, seq_len, num_outcomes) tensor
        mask_3d: (batch_size, seq_len, num_outcomes) boolean tensor
        position_mask_2d: (batch_size, seq_len) boolean tensor indicating which
                          positions to include (before outcome-wise masking).

    Returns:
        X_2d: (n_selected, input_dim) numpy array
        y_2d: (n_selected, num_outcomes) numpy array
    """
    if (
        X_3d.dim() != 3
        or y_3d.dim() != 3
        or mask_3d.dim() != 3
        or position_mask_2d.dim() != 2
    ):
        raise ValueError(
            "flatten_with_custom_mask expects X_3d, y_3d, mask_3d to be 3D "
            "and position_mask_2d to be 2D."
        )

    batch_size, seq_len, _ = X_3d.shape
    if position_mask_2d.shape != (batch_size, seq_len):
        raise ValueError(
            f"position_mask_2d shape {tuple(position_mask_2d.shape)} is incompatible "
            f"with X_3d shape {tuple(X_3d.shape)}"
        )

    # Start from requested positions, then enforce outcome validity
    base_mask = position_mask_2d & mask_3d.any(dim=-1)  # (B, T)
    n_selected = base_mask.sum().item()

    if n_selected == 0:
        # Return empty arrays with correct feature dimensions
        input_dim = X_3d.shape[-1]
        num_outcomes = y_3d.shape[-1]
        return np.empty((0, input_dim)), np.empty((0, num_outcomes))

    X_selected = X_3d[base_mask]  # (n_selected, input_dim)
    y_selected = y_3d[base_mask]  # (n_selected, num_outcomes)

    return X_selected.cpu().numpy(), y_selected.cpu().numpy()


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
        self.do_shift = getattr(args, 'do_shift', False)
        self.interpolation = getattr(args, 'interpolated_output', False)
        self.processor = TimeShiftProcessor(do_shift=do_shift, interpolation=interpolation)
        
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
        print(f"[DEBUG] SklearnModule.fit - After _select_features:")
        print(f"  X_3d shape: {X_3d.shape}, y_3d shape: {y_3d.shape}, mask_3d shape: {mask_3d.shape}")

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
        
        # When do_shift, train on changes (shifted/diffed labels); metrics and storage use levels
        y_3d_for_fit = self.processor.shift_labels(y_3d) if self.do_shift else y_3d
        
        # Reshape for sklearn
        print(f"[DEBUG] SklearnModule.fit - Before reshape_for_sklearn")
        X_2d, y_2d, valid_indices = reshape_for_sklearn(X_3d, y_3d_for_fit, mask_3d)
        
        print(f"Training on {X_2d.shape[0]} valid samples...")
        print(f"[DEBUG] SklearnModule.fit - Final shapes for sklearn:")
        print(f"  X_2d shape: {X_2d.shape}, y_2d shape: {y_2d.shape}")
        
        # Fit sklearn model
        self.model.fit(X_2d, y_2d)
        
        # Compute training metrics (predictions are changes when do_shift; reshift to levels for metrics/storage)
        train_preds_2d = self.model.predict(X_2d)
        train_preds_3d = reconstruct_from_sklearn(train_preds_2d, y_3d.shape, valid_indices, mask_3d)
        if self.do_shift:
            train_preds_3d = self.processor.reshift_labels(train_preds_3d, y_3d, mask_3d)

        # Metadata from batch_dict
        seq_ids = batch_dict.get('seq_id', batch_dict.get('seq_idx', None))
        time_ids = batch_dict.get('time_ids', None)
        oots_mask = batch_dict.get('oots_mask', None)
        ooss_mask = batch_dict.get('ooss_mask', None)
        
        # Store predictions (level-scale)
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
        print(f"[DEBUG] SklearnModule.evaluate - After _select_features:")
        print(f"  X_3d shape: {X_3d.shape}, y_3d shape: {y_3d.shape}, mask_3d shape: {mask_3d.shape}")

        # Reshape for sklearn
        print(f"[DEBUG] SklearnModule.evaluate - Before reshape_for_sklearn")
        X_2d, y_2d, valid_indices = reshape_for_sklearn(X_3d, y_3d, mask_3d)
        
        print(f"Evaluating on {X_2d.shape[0]} valid samples...")
        print(f"[DEBUG] SklearnModule.evaluate - Final shapes for sklearn:")
        print(f"  X_2d shape: {X_2d.shape}, y_2d shape: {y_2d.shape}")
        
        # Predict (model outputs changes when do_shift; reshift to levels for metrics/storage)
        preds_2d = self.model.predict(X_2d)
        preds_3d = reconstruct_from_sklearn(preds_2d, y_3d.shape, valid_indices, mask_3d)
        # do_shift = getattr(self.args, 'do_shift', False)
        if self.do_shift:
            preds_3d = self.processor.reshift_labels(preds_3d, y_3d, mask_3d)

        seq_ids = batch_dict.get('seq_id', batch_dict.get('seq_idx', None))
        time_ids = batch_dict.get('time_ids', None)
        oots_mask = batch_dict.get('oots_mask', None)
        ooss_mask = batch_dict.get('ooss_mask', None)
        
        # Store predictions (level-scale)
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
        When do_shift is True and labels are present, returned predictions
        are on the level scale (reshifted). When do_shift is True and labels
        are absent, returned predictions are change predictions (raw).
        
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
        if self.do_shift and y is not None and mask is not None:
            preds_3d = self.processor.reshift_labels(preds_3d, y, mask)
        
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


def print_split_statistics(X_3d: torch.Tensor, y_3d: torch.Tensor, mask_3d: torch.Tensor, split_name: str = 'train'):
    """
    Print diagnostic statistics for a data split.
    
    Args:
        X_3d: (batch_size, seq_len, input_dim) tensor
        y_3d: (batch_size, seq_len, num_outcomes) tensor
        mask_3d: (batch_size, seq_len, num_outcomes) boolean tensor
        split_name: Name of the split (e.g., 'train', 'test', 'val')
    """
    batch_size = X_3d.shape[0]
    seq_len = X_3d.shape[1]
    
    # Determine which timesteps have at least one valid outcome
    valid_time = mask_3d.any(dim=-1)  # (B, T)
    sequences_with_valid_time = valid_time.any(dim=1).sum().item()
    
    # Count total positions (seq-timesteps)
    total_positions = batch_size * seq_len
    
    # Count valid positions (valid seq-timesteps)
    valid_positions = valid_time.sum().item()
    
    # Extract all valid outcome values
    valid_outcomes_list = []
    for s in range(batch_size):
        for t in range(seq_len):
            if valid_time[s, t]:
                # This position has at least one valid outcome
                # Extract all valid outcomes at this position
                for o in range(y_3d.shape[2]):
                    if mask_3d[s, t, o]:
                        valid_outcomes_list.append(float(y_3d[s, t, o].item()))
    
    if len(valid_outcomes_list) == 0:
        mean_outcome = float('nan')
        std_outcome = float('nan')
    else:
        valid_outcomes_array = np.array(valid_outcomes_list)
        mean_outcome = float(np.mean(valid_outcomes_array))
        std_outcome = float(np.std(valid_outcomes_array))
    
    print("\n" + "=" * 60)
    print(f"{split_name.capitalize()} Split Statistics:")
    print("=" * 60)
    print(f"Total sequences: {batch_size}")
    print(f"Sequences with at least one valid timestep: {sequences_with_valid_time}")
    print(f"Total positions: {total_positions}")
    print(f"Valid positions: {valid_positions}")
    print(f"Mean outcome: {mean_outcome:.4f}")
    print(f"Std outcome: {std_outcome:.4f}")
    print("=" * 60 + "\n")


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
        
        # Print train split statistics
        train_batch_dict = collect_batch_dict(train_dataloader)
        X_train_3d, y_train_3d, mask_train_3d = module._select_features(train_batch_dict)
        print_split_statistics(X_train_3d, y_train_3d, mask_train_3d, split_name='train')
        
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
        
        # Print validation split statistics
        val_batch_dict = collect_batch_dict(val_dataloader)
        X_val_3d, y_val_3d, mask_val_3d = module._select_features(val_batch_dict)
        print_split_statistics(X_val_3d, y_val_3d, mask_val_3d, split_name='val')
        
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
        
        # Print test split statistics
        test_batch_dict = collect_batch_dict(test_dataloader)
        X_test_3d, y_test_3d, mask_test_3d = module._select_features(test_batch_dict)
        print_split_statistics(X_test_3d, y_test_3d, mask_test_3d, split_name='test')
        
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
                             val_dataloader, search_type='grid', n_iter=10, cv=3, stratify=True):
        """
        Perform hyperparameter search using a custom sequence- and time-aware CV.

        Cross-sectional + prospective CV:
          - Sequences (documents) are split into `cv` folds.
            - If stratify=True: sequences are stratified by outcome mean/std (computed
              only on valid timesteps), then assigned using round-robin.
            - If stratify=False: sequences are split into `cv` contiguous blocks.
          - Global time cutoff: first 2/3 of sequence length are train-time,
            last 1/3 are eval-time (applied uniformly to ALL sequences).
            Only valid timesteps (where outcomes_mask is True) are considered.
          - For fold k (holding out sequences S_k):
              * Train:
                  - sequences in S_train = all_sequences \\ S_k
                  - timesteps in train-time (first 2/3 of sequence length, where valid)
              * Eval (union):
                  - longitudinal eval (fixed across folds):
                        sequences in S_train, timesteps in eval-time (last 1/3 of sequence length, where valid)
                  - within-time OOSS:
                        sequences in S_k, timesteps in train-time (first 2/3 of sequence length, where valid)
        
        Args:
            module: SklearnModule instance (will be cloned)
            param_grid: Dict of parameter names to lists of values
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            search_type: 'grid' or 'random'
            n_iter: Number of iterations for random search
            cv: Number of cross-validation folds
            stratify: If True, stratify sequences by outcome mean/std; if False, use contiguous blocks
            
        Returns:
            SklearnModule with best model
        """
        print("=" * 60)
        stratify_str = "stratified" if stratify else "contiguous"
        print(f"Hyperparameter search ({search_type}) with custom sequence/time CV ({stratify_str})...")
        print("=" * 60)
        
        # Extract training data using adapter pattern
        train_batch_dict = collect_batch_dict(train_dataloader)
        X_train_3d, y_train_3d, mask_train_3d = module._select_features(train_batch_dict)
        y_train_3d_for_fit = module.processor.shift_labels(y_train_3d) if module.do_shift else y_train_3d

        # Build custom CV masks over sequences and time
        cv_masks = build_sequence_time_cv_masks(
            X_train_3d, y_train_3d, outcomes_mask_3d=mask_train_3d, n_folds=cv, stratify=stratify
        )

        # Print fold statistics for debugging stratification
        print("\n" + "=" * 60)
        print("Fold Statistics (before hyperparameter search):")
        print("=" * 60)
        
        # First, get overall statistics
        batch_size = y_train_3d.shape[0]
        
        # Debug: Check mask structure
        print(f"[DEBUG] Mask shape: {mask_train_3d.shape}")
        print(f"[DEBUG] Mask dtype: {mask_train_3d.dtype}")
        print(f"[DEBUG] Mask sum (total True values): {mask_train_3d.sum().item()}")
        print(f"[DEBUG] Mask any per timestep shape: {mask_train_3d.any(dim=-1).shape}")
        
        valid_time = mask_train_3d.any(dim=-1)  # (B, T)
        sequences_with_valid_time = valid_time.any(dim=1).sum().item()
        
        # Also check if sequences have any data at all (check pad_mask if available)
        if 'pad_mask' in train_batch_dict:
            pad_mask = train_batch_dict['pad_mask']  # (B, T)
            sequences_with_data = pad_mask.any(dim=1).sum().item()
            print(f"[DEBUG] Sequences with any data (from pad_mask): {sequences_with_data}")
            print(f"[DEBUG] pad_mask shape: {pad_mask.shape}, dtype: {pad_mask.dtype}")
        
        print(f"Total sequences in dataset: {batch_size}")
        print(f"Sequences with at least one valid timestep (from outcomes_mask): {sequences_with_valid_time}\n")
        
        for fold_idx, (train_mask_2d, eval_mask_2d, train_mask_2d_unfiltered, eval_mask_2d_unfiltered) in enumerate(cv_masks):
            # Extract valid outcomes in eval set
            # Use same logic as flatten_with_custom_mask: position_mask & outcome validity
            base_mask = eval_mask_2d & mask_train_3d.any(dim=-1)  # (B, T)
            
            # Count total positions in eval set (including invalid ones) - use unfiltered mask
            total_n = eval_mask_2d_unfiltered.numel()
            
            # Count sequences that have at least one position in eval set
            sequences_in_eval = (eval_mask_2d.any(dim=1)).sum().item()
            
            # Count sequences with valid timesteps but no eval positions
            sequences_with_eval = eval_mask_2d.any(dim=1)  # (B,)
            sequences_missing_eval = (valid_time.any(dim=1) & ~sequences_with_eval).sum().item()
            
            # Count sequences with no valid timesteps at all
            sequences_no_valid_time = (~valid_time.any(dim=1)).sum().item()
            
            # Count sequences that are fully held out for eval (have eval positions but no train positions)
            # These are the sequences in the held-out fold (OOSS evaluation)
            sequences_with_train = train_mask_2d.any(dim=1)  # (B,)
            sequences_held_out = (sequences_with_eval & ~sequences_with_train).sum().item()
            
            # Extract all valid outcome values from eval set
            valid_outcomes_list = []
            for s in range(y_train_3d.shape[0]):
                for t in range(y_train_3d.shape[1]):
                    if base_mask[s, t]:
                        # This position is in eval set and has at least one valid outcome
                        # Extract all valid outcomes at this position
                        for o in range(y_train_3d.shape[2]):
                            if mask_train_3d[s, t, o]:
                                valid_outcomes_list.append(float(y_train_3d[s, t, o].item()))
            
            if len(valid_outcomes_list) == 0:
                mean_val = float('nan')
                std_val = float('nan')
                valid_n = 0
            else:
                valid_outcomes_array = np.array(valid_outcomes_list)
                mean_val = float(np.mean(valid_outcomes_array))
                std_val = float(np.std(valid_outcomes_array))
                valid_n = len(valid_outcomes_list)
            
            print(f"Fold {fold_idx + 1}/{len(cv_masks)}:")
            print(f"  Sequences in eval set: {sequences_in_eval}")
            print(f"  Sequences fully held out for eval (OOSS): {sequences_held_out}")
            print(f"  Sequences with valid time but missing from eval: {sequences_missing_eval}")
            print(f"  Sequences with no valid timesteps: {sequences_no_valid_time}")
            print(f"  Total N: {total_n} positions")
            print(f"  Valid N: {valid_n} valid samples")
            print(f"  Mean PCL score: {mean_val:.4f}")
            print(f"  Std PCL score: {std_val:.4f}")
        print("=" * 60 + "\n")

        # Prepare hyperparameter combinations
        if search_type == 'grid':
            param_iterable = list(ParameterGrid(param_grid))
        elif search_type == 'random':
            param_iterable = list(
                ParameterSampler(
                    param_distributions=param_grid,
                    n_iter=n_iter,
                    random_state=getattr(module.args, 'seed', 42),
                )
            )
        else:
            raise ValueError(f"Invalid search_type: {search_type}. Use 'grid' or 'random'.")

        print(f"Searching over {len(param_iterable)} hyperparameter combinations...")

        # Track best params per fold (for mode-based selection)
        fold_best_params: List[Optional[Dict[str, Any]]] = [None] * len(cv_masks)
        fold_best_mses: List[Optional[float]] = [None] * len(cv_masks)

        # Loop over hyperparameter combinations
        for combo_idx, params in enumerate(param_iterable):
            print(f"\n--- Hyperparameter set {combo_idx + 1}/{len(param_iterable)}: {params} ---")

            # Cross-validation over folds
            for fold_idx, (train_mask_2d, eval_mask_2d, _, _) in enumerate(cv_masks):
                print(f"  Fold {fold_idx + 1}/{len(cv_masks)}")

                # Train on changes when do_shift; eval targets stay levels
                X_train_2d, y_train_2d = flatten_with_custom_mask(
                    X_train_3d, y_train_3d_for_fit, mask_train_3d, train_mask_2d
                )
                X_eval_2d, y_eval_2d = flatten_with_custom_mask(
                    X_train_3d, y_train_3d, mask_train_3d, eval_mask_2d
                )

                if X_train_2d.shape[0] == 0 or X_eval_2d.shape[0] == 0:
                    print("    Skipping fold due to empty train/eval set.")
                    continue

                # Clone base model and set hyperparameters
                model = deepcopy(module.model)
                if params:
                    model.set_params(**params)

                # Fit and evaluate
                model.fit(X_train_2d, y_train_2d)
                preds_eval = model.predict(X_eval_2d)
                # Ensure 2D for comparison
                preds_eval = np.asarray(preds_eval)
                if preds_eval.ndim == 1:
                    preds_eval = preds_eval.reshape(-1, 1)

                if module.do_shift:
                    # Model predicts changes; reshift to levels before MSE vs y_eval_2d (levels)
                    base_mask = eval_mask_2d & mask_train_3d.any(dim=-1)
                    preds_eval_3d = torch.zeros_like(y_train_3d)
                    preds_eval_3d[base_mask] = torch.from_numpy(preds_eval).to(
                        device=y_train_3d.device, dtype=y_train_3d.dtype
                    )
                    preds_eval_3d = module.processor.reshift_labels(preds_eval_3d, y_train_3d, mask_train_3d)
                    preds_eval_levels = preds_eval_3d[base_mask].cpu().numpy()
                    mse = np.mean((preds_eval_levels - y_eval_2d) ** 2)
                else:
                    mse = np.mean((preds_eval - y_eval_2d) ** 2)
                print(f"    Fold {fold_idx + 1} MSE: {mse:.6f}")

                # Update best params for this fold if MSE improved
                if fold_best_mses[fold_idx] is None or mse < fold_best_mses[fold_idx]:
                    fold_best_mses[fold_idx] = mse
                    fold_best_params[fold_idx] = params.copy()
                    print(f"    -> New best for fold {fold_idx + 1}")

        # Filter out folds that had no valid evaluations
        valid_fold_indices = [i for i in range(len(fold_best_params)) if fold_best_params[i] is not None]
        if not valid_fold_indices:
            raise RuntimeError("Hyperparameter search failed: no valid hyperparameter set found.")

        # Extract alpha values from best params for each fold
        fold_best_alphas = []
        for fold_idx in valid_fold_indices:
            alpha = fold_best_params[fold_idx].get('alpha', None)
            if alpha is not None:
                fold_best_alphas.append(alpha)

        if not fold_best_alphas:
            # No alpha found, fall back to using first fold's best params
            print("Warning: No 'alpha' parameter found. Using first valid fold's best params.")
            best_params = fold_best_params[valid_fold_indices[0]].copy()
            best_cv_mse = float(np.mean([fold_best_mses[i] for i in valid_fold_indices]))
        else:
            # Find mode alpha (most frequent across folds)
            from collections import Counter
            alpha_counts = Counter(fold_best_alphas)
            max_count = max(alpha_counts.values())
            mode_alphas = [alpha for alpha, count in alpha_counts.items() if count == max_count]

            if len(mode_alphas) == 1:
                # Unique mode
                best_alpha = mode_alphas[0]
            else:
                # Tie: pick highest alpha (most regularization)
                best_alpha = max(mode_alphas)
                print(f"  Tie in mode alpha selection: {mode_alphas}. Choosing highest: {best_alpha}")

            # Find the params dict that matches best_alpha
            best_params = None
            for fold_idx in valid_fold_indices:
                if fold_best_params[fold_idx].get('alpha') == best_alpha:
                    best_params = fold_best_params[fold_idx].copy()
                    break

            if best_params is None:
                # Fallback: use first fold's best params
                print("Warning: Could not find params matching mode alpha. Using first valid fold's best params.")
                best_params = fold_best_params[valid_fold_indices[0]].copy()

            # Compute mean MSE across all folds for logging
            best_cv_mse = float(np.mean([fold_best_mses[i] for i in valid_fold_indices]))

        print(f"\nBest parameters (mode alpha across folds): {best_params}")
        print(f"Best alpha: {best_params.get('alpha', 'N/A')}")
        print(f"Mean CV MSE across folds: {best_cv_mse:.4f}")
        if fold_best_alphas:
            print(f"Fold-wise best alphas: {fold_best_alphas}")

        # Log best parameters
        if self.logger:
            self.logger.log_hyperparams(best_params)
            self.logger.log_metrics({'cv_best_mse': best_cv_mse}, step=0)

        # ---- Refit best model on full training data (first 2/3 timesteps per sequence) ----
        batch_size, seq_len, _ = X_train_3d.shape
        valid_time = mask_train_3d.any(dim=-1)  # (B, T)
        full_train_mask_2d = torch.zeros(batch_size, seq_len, dtype=torch.bool)

        for s in range(batch_size):
            vt = valid_time[s]
            valid_positions = torch.nonzero(vt, as_tuple=False).squeeze(-1)
            n_valid = valid_positions.numel()
            if n_valid == 0:
                continue
            cut = int(np.floor(2.0 * n_valid / 3.0))
            if cut <= 0:
                # If no train-time positions, skip this sequence for training
                continue
            train_positions = valid_positions[:cut]
            full_train_mask_2d[s, train_positions] = True

        full_train_mask_2d = full_train_mask_2d & valid_time

        X_full_train_2d, y_full_train_2d = flatten_with_custom_mask(
            X_train_3d, y_train_3d_for_fit, mask_train_3d, full_train_mask_2d
        )

        if X_full_train_2d.shape[0] == 0:
            raise RuntimeError("No valid training samples found when refitting best model.")

        best_estimator = deepcopy(module.model)
        if best_params:
            best_estimator.set_params(**best_params)
        best_estimator.fit(X_full_train_2d, y_full_train_2d)

        # Extract validation data for final evaluation using adapter pattern
        val_batch_dict = collect_batch_dict(val_dataloader)
        X_val_3d, y_val_3d, mask_val_3d = module._select_features(val_batch_dict)
        X_val_2d, y_val_2d, valid_indices_val = reshape_for_sklearn(
            X_val_3d, y_val_3d, mask_val_3d
        )
        
        # Extract additional metadata from validation batch dict
        seq_ids_val = val_batch_dict.get('seq_id', val_batch_dict.get('seq_idx', None))
        time_ids_val = val_batch_dict.get('time_ids', None)
        oots_val = val_batch_dict.get('oots_mask', None)
        ooss_val = val_batch_dict.get('ooss_mask', None)

        # Create new module with best model
        best_module = SklearnModule(module.args, best_estimator)

        # Evaluate on validation set (model outputs changes when do_shift; reshift to levels for metrics/storage)
        val_preds_2d = best_estimator.predict(X_val_2d)
        val_preds_3d = reconstruct_from_sklearn(
            val_preds_2d, y_val_3d.shape, valid_indices_val, mask_val_3d
        )
        if module.do_shift:
            val_preds_3d = best_module.processor.reshift_labels(val_preds_3d, y_val_3d, mask_val_3d)
        
        # Store predictions (level-scale)
        best_module.predictions['val'] = {
            'preds': val_preds_3d,
            'outcomes': y_val_3d,
            'outcomes_mask': mask_val_3d,
            'seq_id': seq_ids_val,
            'time_ids': time_ids_val,
            'oots_mask': oots_val,
            'ooss_mask': ooss_val
        }
        
        # Compute validation metrics (exhaustive)
        val_metrics = best_module.compute_exhaustive_metrics(
            val_preds_3d, y_val_3d, mask_val_3d, oots_val, ooss_val
        )
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
