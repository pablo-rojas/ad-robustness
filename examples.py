import torch
import numpy as np
from sklearn.metrics import roc_curve, auc

def partial_auc(anomaly_free_scores, anomalous_scores, fpr_threshold=0.2):
    """
    Calculate the partial AUC up to a given false positive rate (FPR) threshold.
    
    Args:
        anomaly_free_scores (list or np.array): Anomaly scores for normal images
        anomalous_scores (list or np.array): Anomaly scores for anomalous images
        fpr_threshold (float): Maximum false positive rate to consider for the partial AUC
    
    Returns:
        float: The partial AUC value up to the specified FPR threshold
    """
    # Ensure inputs are numpy arrays and combine scores
    anomaly_free_scores = np.array(anomaly_free_scores)
    anomalous_scores = np.array(anomalous_scores)
    y_scores = np.concatenate([anomaly_free_scores, anomalous_scores])
    y_true = np.concatenate([np.zeros(len(anomaly_free_scores)), np.ones(len(anomalous_scores))])
    
    # Calculate the ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    
    # Select points where FPR is below the threshold
    valid_idx = np.where(fpr <= fpr_threshold)[0]
    fpr_restricted = fpr[valid_idx]
    tpr_restricted = tpr[valid_idx]
    
    # If the last point is below fpr_threshold, add the point at fpr_threshold via linear interpolation
    if fpr_restricted[-1] < fpr_threshold:
        # Find index of the first FPR value above the threshold
        idx = np.searchsorted(fpr, fpr_threshold)
        # Ensure we don't go out of bounds
        if idx < len(fpr):
            # Linear interpolation for TPR at fpr_threshold
            fpr_low, fpr_high = fpr[idx - 1], fpr[idx]
            tpr_low, tpr_high = tpr[idx - 1], tpr[idx]
            tpr_interp = tpr_low + (tpr_high - tpr_low) * ((fpr_threshold - fpr_low) / (fpr_high - fpr_low))
            
            # Append the interpolated point
            fpr_restricted = np.append(fpr_restricted, fpr_threshold)
            tpr_restricted = np.append(tpr_restricted, tpr_interp)
    
    # Compute the partial AUC using trapezoidal integration
    pAUC = auc(fpr_restricted, tpr_restricted)
    
    return pAUC

def sd_statistic(discriminator_output, target_label):
    """
    Converts discriminator output into a single statistic for anomaly detection.
    
    The function changes the sign of the output to be positive and converts 
    all negative results to 100 (a very large number).

    Args:
        discriminator_output (tuple): Tuple containing (probability tensor, output tensor)
        target_label (int): Target label of the attack
        
    Returns:
        torch.Tensor: The anomaly detection statistic
    """
    aux_prob, aux_out = discriminator_output
    s_d = torch.log(aux_prob) + torch.log(aux_out[:, target_label])
    return -s_d

# Note how my pAUC assumes that the anomaly score will be larger for the anomalous images
# than for the normal images. As with your model this is not the case, I have to change the sign of the output
# to get the correct result.

as_ACGAN = sd_statistic(gan.discriminator(adv_images), target_labels)