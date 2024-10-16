import numpy as np
import time


def np_cross_entropy_loss(preds, targets, reduction='none'):
    """
    Compute cross-entropy loss using precomputed softmax-ed predictions.
    
    Args:
        preds: Numpy array of softmax-ed predictions (shape: num_samples, num_classes).
        targets: Numpy array of true class indices (shape: num_samples,).
        reduction: Reduction method ('none', 'mean', or 'sum').
    
    Returns:
        Cross-entropy loss (either per-sample or aggregated).
    """
    # Ensure targets are integers (as index arrays must be integers)
    targets = np.asarray(targets, dtype=np.int32)

    # Efficient log probabilities calculation
    print("Computing log probabilities for true class...")
    log_prob_start = time.time()
    
    # Compute the log probabilities for the true class
    log_preds = np.log(preds[np.arange(len(targets)), targets] + 1e-9)

    log_prob_end = time.time()
    print(f"Log probabilities calculation took {log_prob_end - log_prob_start:.4f} seconds")

    # Cross-entropy loss (negative log likelihood)
    print("Computing cross-entropy loss...")
    loss_start = time.time()
    loss = -log_preds  # In-place loss calculation
    loss_end = time.time()
    print(f"Cross-entropy loss calculation took {loss_end - loss_start:.4f} seconds")

    # Apply reduction (if any)
    if reduction == 'mean':
        return np.mean(loss)
    elif reduction == 'sum':
        return np.sum(loss)
    else:
        return loss  # No reduction, return per-sample loss


def np_accuracy(preds, targets):
    """
    Compute accuracy from softmax-ed predictions.
    
    Args:
        preds: Numpy array of softmax-ed predictions (shape: num_samples, num_classes).
        targets: Numpy array of true class indices (shape: num_samples,).
    
    Returns:
        Accuracy as a float.
    """
    pred_classes = np.argmax(preds, axis=1)
    return np.mean(pred_classes == targets)

def np_macro_accuracy(preds, targets, num_classes):
    """
    Compute macro accuracy using precomputed softmax-ed predictions.
    
    Args:
        preds: Numpy array of softmax-ed predictions (shape: num_samples, num_classes).
        targets: Numpy array of true class indices (shape: num_samples,).
        num_classes: The total number of classes.
    
    Returns:
        Macro accuracy as a float.
    """
    pred_classes = np.argmax(preds, axis=1)  # Get predicted classes (shape: num_samples)
    
    class_accuracies = []
    for c in range(num_classes):
        # Find indices of true labels that belong to class `c`
        class_idxs = np.where(targets == c)[0]
        
        # If there are no instances of this class in the targets, skip it
        if len(class_idxs) == 0:
            continue
        
        # Calculate accuracy for this class
        class_correct = np.sum(pred_classes[class_idxs] == targets[class_idxs])
        class_accuracy = class_correct / len(class_idxs)
        
        class_accuracies.append(class_accuracy)
    
    # Compute macro accuracy (average per-class accuracy)
    macro_accuracy = np.mean(class_accuracies)
    
    return macro_accuracy
