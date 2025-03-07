import numpy as np
from typing import Dict, Any

def calculate_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """
    Calculate evaluation metrics for predictions.
    
    Args:
        predictions: Predicted values (batch_size, 14)
        targets: Target values (batch_size, 14)
    
    Returns:
        Dictionary of metrics
    """
    # Ensure predictions and targets have the same shape
    if predictions.shape != targets.shape:
        raise ValueError(f"Shapes of predictions {predictions.shape} and targets {targets.shape} do not match")
    
    # Mean Absolute Error
    mae = np.mean(np.abs(predictions - targets))
    
    # Root Mean Squared Error
    rmse = np.sqrt(np.mean(np.square(predictions - targets)))
    
    # Mean Absolute Percentage Error
    # Avoid division by zero
    mask = targets != 0
    if np.any(mask):
        mape = np.mean(np.abs((targets[mask] - predictions[mask]) / targets[mask])) * 100
    else:
        mape = 0.0
    
    # Directional Accuracy
    # For each day, check if the model correctly predicts the direction of price movement
    # We need to compare predictions[i] with targets[i-1] to check direction
    direction_correct = 0
    total_directions = 0
    
    # For each instance in the batch
    for i in range(predictions.shape[0]):
        # For each day in the prediction (7 days)
        for j in range(7):
            # Skip the first day as we need previous day's close
            if j > 0:
                # Index for the previous day's close and current day's close in the flattened array
                prev_day_close_idx = j * 2 - 1  # Previous day's close (odd indices)
                curr_day_close_idx = j * 2 + 1  # Current day's close (odd indices)
                
                # Get predicted and actual direction
                pred_direction = predictions[i, curr_day_close_idx] - predictions[i, prev_day_close_idx]
                true_direction = targets[i, curr_day_close_idx] - targets[i, prev_day_close_idx]
                
                # Check if directions match (both positive or both negative)
                if (pred_direction * true_direction) > 0:
                    direction_correct += 1
                
                total_directions += 1
    
    dir_acc = direction_correct / total_directions * 100 if total_directions > 0 else 0
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'dir_acc': dir_acc
    }