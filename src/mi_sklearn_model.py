"""
Sklearn model adapter classes for use with SklearnModule and SklearnTrainer.

These classes implement the `select_features()` interface that allows sklearn
models to work with PyTorch DataLoaders by extracting and shaping features
from batch dictionaries.
"""
from typing import Dict, Tuple
import torch
from sklearn.linear_model import Ridge, Lasso


class RidgeForecastModel(Ridge):
    """
    Example sklearn model adapter that knows how to select features
    from the aggregated batch dictionary.

    It inherits from sklearn's Ridge so it can be used directly with
    GridSearchCV / RandomizedSearchCV, but also implements:

        select_features(batch_dict, args) -> (X_3d, y_3d, mask_3d)

    so that SklearnModule can delegate feature selection to the model
    (mirroring the PyTorch design where the model decides which
    embeddings_* keys to use).
    """

    def select_features(self, batch_dict, args):
        """
        Args:
            batch_dict: Dict[str, Tensor] aggregated over all batches.
            args: Namespace with configuration (unused here but kept for API).

        Returns:
            X_3d: (batch_size, seq_len, input_dim)
            y_3d: (batch_size, seq_len, num_outcomes)
            mask_3d: (batch_size, seq_len, num_outcomes) boolean
        """
        # Prefer language embeddings if available, otherwise fall back to
        # the first embeddings_* key. Users can customize this logic.
        embeddings_key = None
        preferred_keys = [
            "embeddings_lang_z",
            "embeddings_lang",
            "embeddings_subscales_z",
            "embeddings_subscales",
        ]
        available_embedding_keys = [k for k in batch_dict.keys() if k.startswith("embeddings")]

        for k in preferred_keys:
            if k in batch_dict:
                embeddings_key = k
                break

        if embeddings_key is None:
            if len(available_embedding_keys) == 0:
                raise KeyError(
                    f"No embeddings_* keys found in batch_dict. "
                    f"Available keys: {list(batch_dict.keys())}"
                )
            # Fallback to the first embeddings_* key
            embeddings_key = available_embedding_keys[0]

        if "outcomes" not in batch_dict or "outcomes_mask" not in batch_dict:
            raise KeyError(
                "Expected 'outcomes' and 'outcomes_mask' in batch_dict. "
                f"Available keys: {list(batch_dict.keys())}"
            )

        print(f"Using embeddings key: {embeddings_key}")
        X_3d = batch_dict[embeddings_key]
        y_3d = batch_dict["outcomes"]
        mask_3d = batch_dict["outcomes_mask"]

        return X_3d, y_3d, mask_3d


class LassoForecastModel(Lasso):
    """
    Lasso variant of RidgeForecastModel implementing the same
    select_features interface.
    """

    def select_features(self, batch_dict, args):
        # Reuse the same logic as RidgeForecastModel
        return RidgeForecastModel.select_features(self, batch_dict, args)


# ============================================================================
# Autoregressive Sklearn Models with History Windowing
# ============================================================================

class AutoRegressiveSklearnBase:
    """
    Base class for autoregressive sklearn models that use history windowing.
    
    This class implements the unfolding/windowing logic in `select_features()`
    similar to `AutoRegressiveLinear2` in `mi_model.py`. It handles:
    - Left padding with zeros (causal padding)
    - Sliding window unfolding
    - Flattening windows to create expanded feature vectors
    - Mask unfolding to properly handle valid windows
    
    Concrete sklearn models (Ridge, Lasso, etc.) should inherit from both
    this class and the sklearn model class using multiple inheritance.
    """
    
    def select_features(self, batch_dict: Dict[str, torch.Tensor], args) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select features with autoregressive history windowing.
        
        This method:
        1. Extracts embeddings from batch_dict
        2. Pads on the left with (max_len - 1) zeros for causal sliding
        3. Unfolds to create sliding windows of size max_len
        4. Flattens windows: (B, T, max_len, D) -> (B, T, max_len * D)
        5. Handles mask unfolding to mark valid windows
        
        Args:
            batch_dict: Dictionary with keys like 'embeddings_*', 'outcomes', 'outcomes_mask', etc.
            args: Namespace with configuration (must have args.max_len for history size)
        
        Returns:
            X_3d: (batch_size, seq_len, max_len * input_dim) tensor - unfolded features
            y_3d: (batch_size, seq_len, num_outcomes) tensor - targets (unchanged)
            mask_3d: (batch_size, seq_len, num_outcomes) tensor - adjusted for valid windows
        """
        # Get max_len from args
        max_len = getattr(args, 'max_len', 1)
        if max_len < 1:
            raise ValueError(f"max_len must be >= 1, got {max_len}")
        
        padding_len = max_len - 1
        
        # Extract embeddings (same logic as RidgeForecastModel)
        embeddings_key = None
        preferred_keys = [
            "embeddings_lang_z",
            "embeddings_lang",
            "embeddings_subscales_z",
            "embeddings_subscales",
        ]
        available_embedding_keys = [k for k in batch_dict.keys() if k.startswith("embeddings")]
        
        for k in preferred_keys:
            if k in batch_dict:
                embeddings_key = k
                break
        
        if embeddings_key is None:
            if len(available_embedding_keys) == 0:
                raise KeyError(
                    f"No embeddings_* keys found in batch_dict. "
                    f"Available keys: {list(batch_dict.keys())}"
                )
            embeddings_key = available_embedding_keys[0]
        
        if "outcomes" not in batch_dict or "outcomes_mask" not in batch_dict:
            raise KeyError(
                "Expected 'outcomes' and 'outcomes_mask' in batch_dict. "
                f"Available keys: {list(batch_dict.keys())}"
            )
        
        print(f"Using embeddings key: {embeddings_key} (with max_len={max_len} history windowing)")
        
        # Extract base tensors
        X_3d = batch_dict[embeddings_key]  # (batch_size, seq_len, input_dim)
        y_3d = batch_dict["outcomes"]      # (batch_size, seq_len, num_outcomes)
        mask_3d = batch_dict["outcomes_mask"]  # (batch_size, seq_len, num_outcomes)
        
        batch_size, seq_len, input_dim = X_3d.shape
        
        # Pad on the left for causal sliding (same as AutoRegressiveLinear2)
        if padding_len > 0:
            # Pad along sequence dimension (dim=1) with zeros on the left
            X_3d_padded = torch.nn.functional.pad(X_3d, (0, 0, padding_len, 0), mode='constant', value=0.0)
            # Pad mask similarly (with False/0 for padded positions)
            mask_3d_padded = torch.nn.functional.pad(mask_3d, (0, 0, padding_len, 0), mode='constant', value=False)
        else:
            X_3d_padded = X_3d
            mask_3d_padded = mask_3d
        
        # Unfold to create sliding windows
        # unfold(dimension=1, size=max_len, step=1) creates windows of size max_len
        # Shape: (batch_size, seq_len, max_len, input_dim)
        X_unfolded = X_3d_padded.unfold(dimension=1, size=max_len, step=1)
        
        # Flatten the window dimension: (batch_size, seq_len, max_len, input_dim) -> (batch_size, seq_len, max_len * input_dim)
        batch_size_unfolded, seq_len_unfolded, window_size, input_dim_unfolded = X_unfolded.shape
        assert window_size == max_len, f"Expected window_size={max_len}, got {window_size}"
        assert input_dim_unfolded == input_dim, f"Expected input_dim={input_dim}, got {input_dim_unfolded}"
        
        X_3d_flattened = X_unfolded.contiguous().view(batch_size_unfolded, seq_len_unfolded, max_len * input_dim)
        
        # Handle mask unfolding
        # For a window to be valid, ALL timesteps in the window must have valid outcomes
        # We need to check if all outcomes in the window are valid
        # mask_3d_padded: (batch_size, seq_len + padding_len, num_outcomes)
        # We unfold it similarly and check if all positions in the window are valid
        
        # Unfold mask: (batch_size, seq_len, max_len, num_outcomes)
        mask_unfolded = mask_3d_padded.unfold(dimension=1, size=max_len, step=1)
        
        # A window is valid only if ALL timesteps in the window have at least one valid outcome
        # For each window, check if all timesteps have mask.any(dim=-1) == True
        # mask_unfolded: (batch_size, seq_len, max_len, num_outcomes)
        # For each window position, check if all timesteps in that window are valid
        window_valid = mask_unfolded.any(dim=-1)  # (batch_size, seq_len, max_len) - True if any outcome is valid at each timestep
        window_valid = window_valid.all(dim=-1)   # (batch_size, seq_len) - True if ALL timesteps in window are valid
        
        # Expand to match num_outcomes: (batch_size, seq_len) -> (batch_size, seq_len, num_outcomes)
        mask_3d_adjusted = window_valid.unsqueeze(-1).expand(-1, -1, y_3d.shape[-1])
        
        # Note: y_3d remains unchanged - we still predict for each timestep
        # The mask adjustment ensures we only train on windows where all history is valid
        
        return X_3d_flattened, y_3d, mask_3d_adjusted


class AutoRegressiveRidge(AutoRegressiveSklearnBase, Ridge):
    """
    Ridge regression model with autoregressive history windowing.
    
    This model uses sliding windows of size `max_len` (from args.max_len) to create
    expanded feature vectors. Each timestep's features include the current and
    previous (max_len - 1) timesteps, flattened into a single vector.
    
    Example:
        If max_len=7 and input_dim=64, each timestep will have features of size
        7 * 64 = 448 (concatenation of 7 days of embeddings).
    
    Usage:
        model = AutoRegressiveRidge(alpha=1.0, random_state=42)
        module = SklearnModule(args, model)  # args.max_len must be set
        trainer = SklearnTrainer(output_dir=..., logger=...)
        results = trainer.fit(module, train_dataloader, val_dataloader)
    """
    pass


class AutoRegressiveLasso(AutoRegressiveSklearnBase, Lasso):
    """
    Lasso regression model with autoregressive history windowing.
    
    This model uses sliding windows of size `max_len` (from args.max_len) to create
    expanded feature vectors. Each timestep's features include the current and
    previous (max_len - 1) timesteps, flattened into a single vector.
    
    Example:
        If max_len=7 and input_dim=64, each timestep will have features of size
        7 * 64 = 448 (concatenation of 7 days of embeddings).
    
    Usage:
        model = AutoRegressiveLasso(alpha=1.0, random_state=42, max_iter=1000)
        module = SklearnModule(args, model)  # args.max_len must be set
        trainer = SklearnTrainer(output_dir=..., logger=...)
        results = trainer.fit(module, train_dataloader, val_dataloader)
    """
    pass

