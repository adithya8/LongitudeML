# This is a script to perform multilevel evaluations. 
# We would be computing within-subject and between-subject evaluations.  
from typing import List, Union
import numpy as np
from scipy.stats import pearsonr, hmean


def compute_mse(label:np.ndarray, pred:np.ndarray, outcome_mask:np.ndarray=None):
    """
        Calculate MSE loss for a single outcome. \n
        MSE = mean((label - pred)^2) \n
        Only computed for valid timesteps denoted by outcome_mask.
        
        Inputs
        --------
        label: np.ndarray of shape (num_timesteps, 1) or (num_timesteps, ) \n
        pred: np.ndarray of shape (num_timesteps, 1) or (num_timesteps, ) \n
        outcome_mask: np.ndarray of shape (num_timesteps, 1) or (num_timesteps, ). Value of 0 indicates invalid timestep.
        
        Returns
        --------
        mse_value: float - MSE value.
    """
    assert label.shape == pred.shape, "Shape mismatch between label ({}) and pred ({}).".format(label.shape, pred.shape)
    assert label.shape == outcome_mask.shape, "Shape mismatch between label ({}) and outcome_mask ({}).".format(label.shape, outcome_mask.shape)
    assert len(label.shape) == 1 or (len(label.shape) == 2 and label.shape[1] == 1), "Invalid shape for label ({}).".format(label.shape)
    assert len(pred.shape) == 1 or (len(pred.shape) == 2 and pred.shape[1] == 1), "Invalid shape for pred ({}).".format(pred.shape)
    assert len(outcome_mask.shape) == 1 or (len(outcome_mask.shape) == 2 and outcome_mask.shape[1] == 1), "Invalid shape for outcome_mask ({}).".format(outcome_mask.shape)
    assert (outcome_mask is not None and np.sum(outcome_mask) >= 1) or outcome_mask is None, "Insufficient valid timesteps for Mean Squared Error."
    
    # Change everything to (N, )
    label = label.reshape(-1, )
    pred = pred.reshape(-1, )
    outcome_mask = outcome_mask.reshape(-1, )
    
    # Calculate MSE
    mse_value = np.mean(np.square(label - pred)[outcome_mask == 1])
    
    return mse_value


def compute_mae(label:np.ndarray, pred:np.ndarray, outcome_mask:np.ndarray=None):
    """
        Calculate MAE loss for a single outcome. \n
        MAE = mean(|label - pred|) \n
        Only computed for valid timesteps denoted by outcome_mask.
        
        Inputs
        --------
        label: np.ndarray of shape (num_timesteps, 1) or (num_timesteps, ) \n
        pred: np.ndarray of shape (num_timesteps, 1) or (num_timesteps, ) \n
        outcome_mask: np.ndarray of shape (num_timesteps, 1) or (num_timesteps, ). Value of 0 indicates invalid timestep.
        
        Returns
        --------
        mae_value: float - MAE value.
    """
    assert label.shape == pred.shape, "Shape mismatch between label ({}) and pred ({}).".format(label.shape, pred.shape)
    assert label.shape == outcome_mask.shape, "Shape mismatch between label ({}) and outcome_mask ({}).".format(label.shape, outcome_mask.shape)
    assert len(label.shape) == 1 or (len(label.shape) == 2 and label.shape[1] == 1), "Invalid shape for label ({}).".format(label.shape)
    assert len(pred.shape) == 1 or (len(pred.shape) == 2 and pred.shape[1] == 1), "Invalid shape for pred ({}).".format(pred.shape)
    assert len(outcome_mask.shape) == 1 or (len(outcome_mask.shape) == 2 and outcome_mask.shape[1] == 1), "Invalid shape for outcome_mask ({}).".format(outcome_mask.shape)
    assert (outcome_mask is not None and np.sum(outcome_mask) >= 1) or outcome_mask is None, "Insufficient valid timesteps for Mean Absolute Error."
    
    # Change everything to (N, )
    label = label.reshape(-1, )
    pred = pred.reshape(-1, )
    outcome_mask = outcome_mask.reshape(-1, )
    
    # Calculate MAE
    mae_value = np.mean(np.abs(label - pred)[outcome_mask == 1])
    
    return mae_value


def compute_pearsonr(label:np.ndarray, pred:np.ndarray, outcome_mask:np.ndarray=None):
    """
        Calculate Pearson correlation coefficient for a single outcome. \n
        r = sum((x - x_mean)*(y - y_mean)) / sqrt(sum((x - x_mean)^2) * sum((y - y_mean)^2)) \n
        Computed only for valid timesteps denoted by outcome_mask.
        
        Inputs
        --------
        label: np.ndarray of shape (num_timesteps, 1) or (num_timesteps, ) \n
        pred: np.ndarray of shape (num_timesteps, 1) or (num_timesteps, ) \n
        outcome_mask: np.ndarray of shape (num_timesteps, 1) or (num_timesteps, ). Value of 0 indicates invalid timestep.
        
        Returns
        --------
        pearson_value: float - Pearson correlation coefficient. \n
        p_value: float - p-value of the correlation.
    """
    assert label.shape == pred.shape, "Shape mismatch between label ({}) and pred ({}).".format(label.shape, pred.shape)
    assert label.shape == outcome_mask.shape, "Shape mismatch between label ({}) and outcome_mask ({}).".format(label.shape, outcome_mask.shape)
    assert len(label.shape) == 1 or (len(label.shape) == 2 and label.shape[1] == 1), "Invalid shape for label ({}).".format(label.shape)
    assert len(pred.shape) == 1 or (len(pred.shape) == 2 and pred.shape[1] == 1), "Invalid shape for pred ({}).".format(pred.shape)
    assert len(outcome_mask.shape) == 1 or (len(outcome_mask.shape) == 2 and outcome_mask.shape[1] == 1), "Invalid shape for outcome_mask ({}).".format(outcome_mask.shape)
    assert (outcome_mask is not None and np.sum(outcome_mask) >= 2) or outcome_mask is None, "Insufficient valid timesteps for Pearson correlation coefficient."
    
    # Change everything to (N, )
    label = label.reshape(-1, )
    pred = pred.reshape(-1, )
    outcome_mask = outcome_mask.reshape(-1, )

    # Calculate Pearson correlation coefficient
    pearson_value, p_value = pearsonr(label[outcome_mask == 1], pred[outcome_mask == 1])
    
    return (pearson_value, p_value)


def compute_smape(label:np.ndarray, pred:np.ndarray, outcome_mask:np.ndarray=None):
    """
        Calculate SMAPE loss for a single outcome. \n
        SMAPE = 2 * mean(|label - pred| / (|label| + |pred| + epsilon)) \n
        Only computed for valid timesteps denoted by outcome_mask.
        
        Inputs
        --------
        label: np.ndarray of shape (num_timesteps, 1) or (num_timesteps, ) \n
        pred: np.ndarray of shape (num_timesteps, 1) or (num_timesteps, ) \n
        outcome_mask: np.ndarray of shape (num_timesteps, 1) or (num_timesteps, ). Value of 0 indicates invalid timestep. \n
        
        Returns 
        --------
        smape_value: float - SMAPE value ranging from 0 to 2.
    """
    assert label.shape == pred.shape, "Shape mismatch between label ({}) and pred ({}).".format(label.shape, pred.shape)
    assert label.shape == outcome_mask.shape, "Shape mismatch between label ({}) and outcome_mask ({}).".format(label.shape, outcome_mask.shape)
    assert len(label.shape) == 1 or (len(label.shape) == 2 and label.shape[1] == 1), "Invalid shape for label ({}).".format(label.shape)
    assert len(pred.shape) == 1 or (len(pred.shape) == 2 and pred.shape[1] == 1), "Invalid shape for pred ({}).".format(pred.shape)
    assert len(outcome_mask.shape) == 1 or (len(outcome_mask.shape) == 2 and outcome_mask.shape[1] == 1), "Invalid shape for outcome_mask ({}).".format(outcome_mask.shape)
    assert (outcome_mask is not None and np.sum(outcome_mask) >= 1) or outcome_mask is None, "Insufficient valid timesteps for SMAPE."
    
    # Change everything to (N, )
    label = label.reshape(-1, )
    pred = pred.reshape(-1, )
    outcome_mask = outcome_mask.reshape(-1, )
    
    # Calculate SMAPE
    epsilon = 1e-8
    nr = np.abs(label[outcome_mask==1] - pred[outcome_mask==1])
    dr = np.abs(label[outcome_mask==1]) + np.abs(pred[outcome_mask==1]) + epsilon
    smape_value = np.mean(nr/dr)*2
    
    return smape_value


def within_seq_metric(metric:Union[str, callable], seq_ids:List, labels:List[List], preds:List[List], outcome_masks:List[List]=None):
    """
        Calculate within-sequence metrics. \n
        metric = mean(metric(label, pred)) \n
        Computes the metric for each sequence and then averages it.
        
        Inputs
        --------
        metric: str or callable - Metric to be computed. \n
        seq_ids: List of shape (num_sequences, ) - Sequence ID \n
        labels: List containing the true labels for each sequence. \n
        preds: List containing the predicted labels for each sequence. \n
        outcome_masks: List containing the outcome masks for each sequence. \n
        
        Returns
        --------
        within_seq_metric_dict:dict - Contains the mean, median, and the metrics for the entire sequence. 
    """
    # Check whether the metric is a callable defined as functions above or is in ["mse", "smape", "pearson"]
    assert (isinstance(metric, str) and metric in ["mse", "smape", "pearsonr", "mae"]) or callable(metric), "Invalid metric. Choose from ['mse', 'smape', 'pearsonr', 'mae'] or provide a callable function."

    if isinstance(metric, str):
        if metric == "mse": 
            metric_func = compute_mse
        elif metric == "smape":
            metric_func = compute_smape
        elif metric == "pearsonr":
            metric_func = compute_pearsonr
        elif metric == "mae":
            metric_func = compute_mae
        metric_name = metric
    else:
        metric_name = metric.__name__.split("_")[-1]
        assert metric_name in ["mse", "smape", "pearsonr", "mae"], "Invalid metric function. Choose from ['mse', 'smape', 'pearsonr', 'mae]"
    
    metric_values_dict = {} # Dictionary to store metric values for each sequence
    mean_metric_value = 0.0
    if metric_name == "pearsonr":
        # Collect pearson values and p-values for each sequence
        # aggregated p_value is the harmonic mean of p-values
        mean_p_values = []
        for _, (seq_id, label, pred, outcome_mask) in enumerate(zip(seq_ids, labels, preds, outcome_masks)):
            pearson_value, p_value = metric_func(label, pred, outcome_mask)
            metric_values_dict[seq_id] = (pearson_value, p_value) if not np.isnan(pearson_value) else (0.0, 1.0)
            mean_metric_value += pearson_value if not np.isnan(pearson_value) else 0.0
            mean_p_values.append(p_value)
        mean_metric_value /= len(labels)
        mean_p_values = hmean(mean_p_values)
    else:
        # Calculate metric for each sequence
        for _, (seq_id, label, pred, outcome_mask) in enumerate(zip(seq_ids, labels, preds, outcome_masks)):
            metric_value = metric_func(label, pred, outcome_mask)
            metric_values_dict[seq_id] = metric_value
            mean_metric_value += metric_value
        mean_metric_value /= len(labels)
        
    # Prepare a dictionary that contains the average, median, and the metrics for the entire sequence along with their seq_id
    median_metric_value = np.median(list(metric_values_dict.values())) if metric_name != "pearsonr" else np.median([x[0] for x in list(metric_values_dict.values())])
    if metric_name == "pearsonr":
        sorted_p_values = sorted([x[1] for x in list(metric_values_dict.values())]) 
        median_p_val = hmean([sorted_p_values[len(sorted_p_values)//2], sorted_p_values[len(sorted_p_values)//2 + 1]]) if len(sorted_p_values)%2 == 0 else sorted_p_values[len(sorted_p_values)//2]
    within_seq_metric_dict = {"mean": mean_metric_value, "median": median_metric_value, "seq_metric_values": metric_values_dict.values(), 'seq_ids': metric_values_dict.keys(), "metric_name": metric_name}
    if metric_name == "pearsonr": 
        within_seq_metric_dict["mean_p_value"] = mean_p_values
        within_seq_metric_dict["median_p_value"] = median_p_val
    
    return within_seq_metric_dict