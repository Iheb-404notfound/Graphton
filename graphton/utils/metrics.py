import torch
import torch.nn.functional as F


def accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute classification accuracy.
    
    Args:
        pred: Predictions [num_samples, num_classes] or [num_samples]
        target: Ground truth labels [num_samples]
        
    Returns:
        Accuracy as a float
    """
    if pred.dim() > 1:
        pred = pred.argmax(dim=-1)
    
    correct = (pred == target).sum().item()
    total = target.size(0)
    
    return correct / total


def f1_score(pred: torch.Tensor, target: torch.Tensor, average: str = 'macro') -> float:
    """
    Compute F1 score.
    
    Args:
        pred: Predictions [num_samples, num_classes] or [num_samples]
        target: Ground truth labels [num_samples]
        average: 'macro', 'micro', or 'weighted'
        
    Returns:
        F1 score as a float
    """
    if pred.dim() > 1:
        pred = pred.argmax(dim=-1)
    
    num_classes = target.max().item() + 1
    
    # Compute per-class metrics
    tp = torch.zeros(num_classes)
    fp = torch.zeros(num_classes)
    fn = torch.zeros(num_classes)
    
    for c in range(num_classes):
        tp[c] = ((pred == c) & (target == c)).sum().item()
        fp[c] = ((pred == c) & (target != c)).sum().item()
        fn[c] = ((pred != c) & (target == c)).sum().item()
    
    # Compute F1 per class
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    
    if average == 'macro':
        return f1.mean().item()
    elif average == 'micro':
        tp_sum = tp.sum()
        fp_sum = fp.sum()
        fn_sum = fn.sum()
        precision = tp_sum / (tp_sum + fp_sum + 1e-10)
        recall = tp_sum / (tp_sum + fn_sum + 1e-10)
        return (2 * precision * recall / (precision + recall + 1e-10)).item()
    elif average == 'weighted':
        weights = (tp + fn) / (tp + fn).sum()
        return (f1 * weights).sum().item()
    else:
        raise ValueError(f"Unknown average method: {average}")


def cross_entropy_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute cross-entropy loss.
    
    Args:
        pred: Predictions [num_samples, num_classes]
        target: Ground truth labels [num_samples]
        
    Returns:
        Loss tensor
    """
    return F.cross_entropy(pred, target)


def mean_squared_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute mean squared error.
    
    Args:
        pred: Predictions [num_samples, ...]
        target: Ground truth [num_samples, ...]
        
    Returns:
        MSE loss
    """
    return F.mse_loss(pred, target)


def mean_absolute_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute mean absolute error.
    
    Args:
        pred: Predictions [num_samples, ...]
        target: Ground truth [num_samples, ...]
        
    Returns:
        MAE as a float
    """
    return (pred - target).abs().mean().item()


def auc_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute ROC AUC score (simplified version).
    
    Args:
        pred: Prediction scores [num_samples]
        target: Binary labels [num_samples]
        
    Returns:
        AUC score
    """
    # Sort by prediction scores
    sorted_indices = torch.argsort(pred, descending=True)
    sorted_target = target[sorted_indices]
    
    # Count positives and negatives
    num_pos = (sorted_target == 1).sum().item()
    num_neg = (sorted_target == 0).sum().item()
    
    if num_pos == 0 or num_neg == 0:
        return 0.5
    
    # Compute AUC using trapezoidal rule
    fp = 0
    tp = 0
    auc = 0.0
    
    for i in range(len(sorted_target)):
        if sorted_target[i] == 1:
            tp += 1
        else:
            fp += 1
            auc += tp
    
    auc /= (num_pos * num_neg)
    
    return auc


class Evaluator:
    """
    Evaluator for graph learning tasks.
    """
    
    def __init__(self, task: str = 'node_classification'):
        """
        Args:
            task: Task type ('node_classification', 'graph_classification', 
                  'link_prediction', 'regression')
        """
        self.task = task
    
    def eval(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        """
        Evaluate predictions.
        
        Args:
            pred: Predictions
            target: Ground truth
            
        Returns:
            Dictionary of metrics
        """
        if self.task == 'node_classification' or self.task == 'graph_classification':
            return {
                'accuracy': accuracy(pred, target),
                'f1_macro': f1_score(pred, target, average='macro'),
            }
        elif self.task == 'link_prediction':
            return {
                'auc': auc_score(pred, target),
            }
        elif self.task == 'regression':
            return {
                'mse': mean_squared_error(pred, target).item(),
                'mae': mean_absolute_error(pred, target),
            }
        else:
            raise ValueError(f"Unknown task: {self.task}")