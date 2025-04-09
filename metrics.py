from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def compute_accuracies(y_true, y_pred_proba):
    accuracies = []
    thresholds = []
    for threshold in np.linspace(0, 1, 100):
        y_pred = (y_pred_proba > threshold).astype(int)
        accuracy = metrics.accuracy_score(y_true, y_pred)
        accuracies.append(accuracy)
        thresholds.append(threshold)
    return accuracies, thresholds

def get_best_accuracy(y_true, y_pred_proba):
    accuracies, thresholds = compute_accuracies(y_true, y_pred_proba)
    best_idx = np.argmax(accuracies)
    return accuracies[best_idx], thresholds[best_idx]

def compute_metrics(y_true, y_pred_proba):
    y_pred = (y_pred_proba.argmax(axis=1)).astype(int)
    accuracy = metrics.accuracy_score(y_true, y_pred) * 100
    if len(set(y_true)) == 2:
        y_pred_proba = y_pred_proba[:, 1] 
        precision = metrics.precision_score(y_true, y_pred) * 100
        recall = metrics.recall_score(y_true, y_pred) * 100
        f1 = metrics.f1_score(y_true, y_pred) * 100
        auc = metrics.roc_auc_score(y_true, y_pred_proba) * 100
        accuracy = round(accuracy, 2)
        precision = round(precision, 2)
        recall = round(recall, 2)
        f1 = round(f1, 2)
        auc = round(auc, 2)
    else:
        precision = None
        recall = None
        f1 = None
        auc = None
    return accuracy, precision, recall, f1, auc

def compute_threshold_metrics(y_true, y_pred_proba, confidence):
    maxes = y_pred_proba.max(axis=1)
    uppers = maxes > confidence
    selected = uppers
    y_true_selected = y_true[selected]
    y_pred_selected = y_pred_proba[selected]
    accuracy, precision, recall, f1, auc = compute_metrics(y_true_selected, y_pred_selected)
    perc_selected = selected.mean()
    return perc_selected, accuracy, precision, recall, f1, auc


def get_threshold_acc(maxes, preds, y_true, threshold):
    selected = maxes > threshold
    if selected.sum() == 0:
        return None
    y_true_selected = y_true[selected]
    y_pred_selected = preds[selected]
    accuracy = metrics.accuracy_score(y_true_selected, y_pred_selected)
    return accuracy

def compute_conformal_quartile(y_true, y_pred_proba, confidence=0.95, low=0.01, high=0.999, epsilon=0.001, quit_none=False, fault_tolerance_quit=0.15):
    assert low <= high
    maxes = y_pred_proba.max(axis=1)
    if maxes.max() < high:
        # pick the high to be the 98th percentile highest in maxes
        high = np.percentile(maxes, 98)
    preds = y_pred_proba.argmax(axis=1)
    low_acc = get_threshold_acc(maxes, preds, y_true, low)
    high_acc = get_threshold_acc(maxes, preds, y_true, high)
    if low_acc is None:
        raise ValueError(f"Accuracy at {low} is None. This means accuracy at {high} is {high_acc}, you should lower both low and high")
    if low_acc >= confidence:
        return low # has overshot
    if high_acc < confidence:
        if high - low < 10 * epsilon:
            if quit_none:
                return None
            gamble = np.quantile(maxes, confidence)
            gamble_acc = get_threshold_acc(maxes, preds, y_true, gamble)
            if gamble_acc is None:
                return None
            if gamble_acc + fault_tolerance_quit >= confidence:
                return gamble
            else:
                return None
        return compute_conformal_quartile(y_true, y_pred_proba, confidence, low, high-10*epsilon, epsilon, quit_none)
    # otherwise high has a higher accuracy and low has a lower accuracy
    mid = (low + high) / 2
    mid_acc = get_threshold_acc(maxes, preds, y_true, mid)
    if mid_acc is None:
        return compute_conformal_quartile(y_true, y_pred_proba, confidence, low, mid, epsilon, quit_none)
    if mid_acc >= confidence: # then the quartile is in between low and mid
        if mid - low < epsilon:
            return mid
        else:
            return compute_conformal_quartile(y_true, y_pred_proba, confidence, low, mid, epsilon, quit_none)
    else:
        if high - mid < epsilon:
            return high
        else:
            return compute_conformal_quartile(y_true, y_pred_proba, confidence, mid, high, epsilon, quit_none)

def compute_conformal_metrics(y_true_val, y_pred_proba_val, y_true_test=None, y_pred_proba_test=None, confidence=0.95, quit_none=False):
    if y_true_test is None or y_pred_proba_test is None:
        y_true_test = y_true_val
        y_pred_proba_test = y_pred_proba_val
    preds = y_pred_proba_test.argmax(axis=1)
    quartile = compute_conformal_quartile(y_true_val, y_pred_proba_val, confidence=confidence, quit_none=quit_none)
    if quartile is None:
        return None, None, None, None, None
    maxes = y_pred_proba_test.max(axis=1)
    selected = maxes > quartile
    perc_selected = selected.mean()
    y_true_selected = y_true_test[selected]
    y_pred_selected = preds[selected]
    val_maxes = y_pred_proba_val.max(axis=1)
    val_preds = y_pred_proba_val.argmax(axis=1)
    val_selected = val_maxes > quartile
    val_y_true_selected = y_true_val[val_selected]
    val_y_pred_selected = val_preds[val_selected]
    return perc_selected, metrics.accuracy_score(y_true_selected, y_pred_selected), metrics.accuracy_score(val_y_true_selected, val_y_pred_selected), quartile, selected

    
