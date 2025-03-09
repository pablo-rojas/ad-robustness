import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def partial_auc(anomaly_free_scores, anomalous_scores, fpr_threshold=0.2):
    """
    Calculate the partial AUC (pAUC) using sklearn's roc_curve up to a given false positive rate threshold,
    with interpolation to ensure the interval [0, fpr_threshold] is fully covered.
    """
    # Convert inputs to numpy arrays and create labels.
    anomaly_free_scores = np.array(anomaly_free_scores)
    anomalous_scores = np.array(anomalous_scores)
    y_scores = np.concatenate([anomaly_free_scores, anomalous_scores])
    y_true = np.concatenate([np.zeros(len(anomaly_free_scores)), np.ones(len(anomalous_scores))])
    
    # Calculate full ROC curve.
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    
    # Restrict to FPR values below the threshold.
    valid_idx = np.where(fpr <= fpr_threshold)[0]
    fpr_restricted = fpr[valid_idx]
    tpr_restricted = tpr[valid_idx]
    
    # If the ROC does not have a point exactly at fpr_threshold, interpolate.
    if fpr_restricted[-1] < fpr_threshold:
        idx = np.searchsorted(fpr, fpr_threshold)
        if idx < len(fpr):
            # Linear interpolation between the two surrounding points.
            fpr_low, fpr_high = fpr[idx - 1], fpr[idx]
            tpr_low, tpr_high = tpr[idx - 1], tpr[idx]
            tpr_interp = tpr_low + (tpr_high - tpr_low) * ((fpr_threshold - fpr_low) / (fpr_high - fpr_low))
            fpr_restricted = np.append(fpr_restricted, fpr_threshold)
            tpr_restricted = np.append(tpr_restricted, tpr_interp)
    
    return auc(fpr_restricted, tpr_restricted)

def get_roc(clean, attack, thresholds):
    """
    Compute detection (TPR) and false alarm (FPR) rates at each threshold.
    
    Parameters:
    - clean: anomaly-free scores.
    - attack: anomalous scores.
    - thresholds: array of thresholds to test.
    
    Returns:
    - detection_rate: array of true positive rates.
    - false_alarm: array of false positive rates.
    """
    detection_rate = []
    false_alarm = []
    for threshold in thresholds:
        dr = 0
        fl = 0
        for i in range(len(clean)):
            if clean[i] > threshold:
                fl += 1
            if attack[i] > threshold:
                dr += 1
        detection_rate.append(dr / len(attack))
        false_alarm.append(fl / len(clean))
    return np.array(detection_rate), np.array(false_alarm)

def partial_auc_custom(false_alarm, detection_rate, fpr_threshold=0.2):
    """
    Compute partial AUC from the custom ROC curve (false_alarm vs detection_rate)
    up to the given FPR threshold, with interpolation if needed.
    """
    # Ensure the ROC points are sorted in increasing order of false_alarm.
    sort_idx = np.argsort(false_alarm)
    false_alarm = false_alarm[sort_idx]
    detection_rate = detection_rate[sort_idx]
    
    # Select points with FPR below the threshold.
    valid_idx = np.where(false_alarm <= fpr_threshold)[0]
    fpr_restricted = false_alarm[valid_idx]
    tpr_restricted = detection_rate[valid_idx]
    
    # If the highest FPR is below the threshold, interpolate at fpr_threshold.
    if fpr_restricted.size == 0 or fpr_restricted[-1] < fpr_threshold:
        idx = np.searchsorted(false_alarm, fpr_threshold)
        if idx < len(false_alarm) and idx > 0:
            fpr_low = false_alarm[idx - 1]
            fpr_high = false_alarm[idx]
            tpr_low = detection_rate[idx - 1]
            tpr_high = detection_rate[idx]
            tpr_interp = tpr_low + (tpr_high - tpr_low) * ((fpr_threshold - fpr_low) / (fpr_high - fpr_low))
            fpr_restricted = np.append(fpr_restricted, fpr_threshold)
            tpr_restricted = np.append(tpr_restricted, tpr_interp)
    
    return auc(fpr_restricted, tpr_restricted)

def pauc_1(anomaly_free_scores, anomalous_scores, fpr_threshold=0.2):
    """
    Calculate the partial AUC (pAUC) using a simpler approach:
    Compute the ROC curve and then restrict to points where FPR is below the threshold,
    without any interpolation.
    """
    # Combine scores and create corresponding labels.
    y_true = np.array([0] * len(anomaly_free_scores) + [1] * len(anomalous_scores))
    y_scores = np.array(anomaly_free_scores + anomalous_scores)
    
    # Calculate the ROC curve.
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    
    # Find the points where the FPR is below the threshold.
    fpr_threshold_indices = np.where(fpr <= fpr_threshold)[0]
    
    # Restrict the FPR and TPR to those points.
    fpr_restricted = fpr[fpr_threshold_indices]
    tpr_restricted = tpr[fpr_threshold_indices]
    
    # Calculate the partial AUC using trapezoidal integration.
    return auc(fpr_restricted, tpr_restricted)

def partial_auc_custom_no_interp(false_alarm, detection_rate, fpr_threshold=0.2):
    """
    Compute partial AUC from the custom ROC curve (false_alarm vs detection_rate)
    up to the given FPR threshold, WITHOUT interpolation.
    
    Parameters:
    - false_alarm (np.array): False positive rates from the custom ROC.
    - detection_rate (np.array): True positive rates from the custom ROC.
    - fpr_threshold (float): Upper limit for the FPR over which the area is computed.
    
    Returns:
    - pAUC (float): The partial AUC computed only over the ROC points with FPR ≤ fpr_threshold.
    
    Note: If no point reaches the threshold, the integration stops at the maximum FPR
    below the threshold, underestimating the area compared to a full [0, fpr_threshold] integration.
    """
    # Sort the ROC points by FPR.
    sort_idx = np.argsort(false_alarm)
    false_alarm = false_alarm[sort_idx]
    detection_rate = detection_rate[sort_idx]
    
    # Restrict to ROC points where FPR is below the threshold.
    valid_idx = np.where(false_alarm <= fpr_threshold)[0]
    fpr_restricted = false_alarm[valid_idx]
    tpr_restricted = detection_rate[valid_idx]
    
    # Use trapezoidal integration (via sklearn's auc) over the restricted points.
    return auc(fpr_restricted, tpr_restricted)


def compute_roc_vectorized(y_true, y_scores):
    """
    Compute the ROC curve using a vectorized approach given y_true and y_scores.
    Assumes y_true and y_scores are numpy arrays.
    
    Returns:
    - fpr: Array of false positive rates.
    - tpr: Array of true positive rates.
    """
    # Sort in descending order so that higher scores (more positive) come first.
    desc_order = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[desc_order]
    
    # Total number of positives and negatives.
    N_pos = np.sum(y_true_sorted)
    N_neg = len(y_true_sorted) - N_pos
    
    # Cumulative true positives and false positives.
    cum_tp = np.cumsum(y_true_sorted)
    cum_fp = np.cumsum(1 - y_true_sorted)
    
    # Normalize to get TPR and FPR.
    tpr = np.concatenate(([0], cum_tp / N_pos))
    fpr = np.concatenate(([0], cum_fp / N_neg))
    
    return fpr, tpr

def partial_auc_vectorized(fpr, tpr, fpr_threshold=0.2, interpolate=True):
    """
    Compute the partial AUC given fpr and tpr arrays.
    
    Parameters:
    - fpr: Array of false positive rates (should be sorted in increasing order).
    - tpr: Array of true positive rates.
    - fpr_threshold: The maximum FPR up to which to integrate.
    - interpolate: Whether to linearly interpolate to exactly fpr_threshold.
    
    Returns:
    - pAUC: The partial AUC value.
    """
    # Find indices where FPR is below or equal to threshold.
    valid_idx = np.where(fpr <= fpr_threshold)[0]
    fpr_valid = fpr[valid_idx]
    tpr_valid = tpr[valid_idx]
    
    # Optionally interpolate at fpr_threshold if needed.
    if interpolate and fpr_valid[-1] < fpr_threshold:
        # Find the first index where fpr exceeds the threshold.
        idx = np.searchsorted(fpr, fpr_threshold)
        if idx < len(fpr) and idx > 0:
            # Linear interpolation.
            fpr_low, fpr_high = fpr[idx - 1], fpr[idx]
            tpr_low, tpr_high = tpr[idx - 1], tpr[idx]
            tpr_interp = tpr_low + (tpr_high - tpr_low) * ((fpr_threshold - fpr_low) / (fpr_high - fpr_low))
            fpr_valid = np.append(fpr_valid, fpr_threshold)
            tpr_valid = np.append(tpr_valid, tpr_interp)
    # Otherwise, integration is only done up to the highest available FPR below threshold.
    return auc(fpr_valid, tpr_valid)

def main():
    # Load your anomaly scores from file.
    results = np.load('./results/benchmark_best/fgsm_inf_0.05/acgan/results.npy', allow_pickle=True).tolist()
    nat_as = results['nat_list']
    adv_as = results['adv_list']
    
    # Use the loaded scores: nat_as as anomaly-free and adv_as as anomalous scores.
    # Create the true labels for sklearn's roc_curve.
    y_true = np.concatenate([np.zeros(len(nat_as)), np.ones(len(adv_as))])
    y_scores = np.concatenate([nat_as, adv_as])

    # Sort y_true and y_scores based on y_scores.
    sorted_indices = np.argsort(y_scores)
    y_true_s = y_true[sorted_indices]
    y_scores_s = y_scores[sorted_indices]

    # Compute the ROC curve using the vectorized approach.
    fpr, tpr = compute_roc_vectorized(y_true_s, y_scores_s)

    # Compute the partial AUC using the vectorized approach.
    pAUC_vectorized = partial_auc_vectorized(fpr, tpr, fpr_threshold=0.2)
    
    
    # Compute ROC curve using sklearn's roc_curve.
    fpr_sklearn, tpr_sklearn, _ = roc_curve(y_true, y_scores)
    
    # Compute partial AUC using the sklearn-based implementation (with interpolation).
    pAUC_sklearn = partial_auc(nat_as, adv_as, fpr_threshold=0.2)
    
    # Compute ROC curve using the custom implementation.
    thresholds_custom = np.linspace(max(y_scores), min(y_scores), 10000)
    tpr_custom, fpr_custom = get_roc(nat_as, adv_as, thresholds_custom)
    
    # Compute partial AUC using the custom ROC (with interpolation).
    pAUC_custom = partial_auc_custom_no_interp(fpr_custom, tpr_custom, fpr_threshold=0.2)
    
    # Compute partial AUC using the simple implementation (pauc_1) without interpolation.
    pAUC_1 = pauc_1(nat_as, adv_as, fpr_threshold=0.2)
    
    # Plot both ROC curves.
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_sklearn, tpr_sklearn, color='blue', marker='x', label='sklearn ROC')
    plt.plot(fpr_custom, tpr_custom, color='red', marker='x', label='custom ROC')
    plt.plot(fpr, tpr, color='green', marker='x', label='vectorized ROC')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('roc_comparison.png', dpi=1200)  # Save figure with high resolution.
    #plt.show()  # Display the figure.
    plt.close()
    
    # Print the partial AUC results.
    print("Partial AUC (sklearn-based with interpolation):", pAUC_sklearn)
    print("Partial AUC (ACGAN with interpolation):", pAUC_custom)
    print("Partial AUC (pauc_1, no interpolation):", pAUC_1)
    print("Partial AUC (vectorized with interpolation):", pAUC_vectorized)

if __name__ == '__main__':
    main()

