
# adapted from https://github.com/mims-harvard/TDC/blob/main/tdc/evaluator.py

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    average_precision_score,
    precision_score,
    recall_score,
    accuracy_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    precision_recall_curve,
    cohen_kappa_score,
    auc,
    roc_curve
)
from scipy import stats


def range_logAUC(true_y, predicted_score, FPR_range=(0.001, 0.1)):
    """
    Author: Yunchao "Lance" Liu (lanceknight26@gmail.com)
    Calculate logAUC in a certain FPR range (default range: [0.001, 0.1]).
    This was used by previous methods [1] and the reason is that only a
    small percentage of samples can be selected for experimental tests in
    consideration of cost. This means only molecules with very high
    predicted score can be worth testing, i.e., the decision
    threshold is high. And the high decision threshold corresponds to the
    left side of the ROC curve, i.e., those FPRs with small values. Also,
    because the threshold cannot be predetermined, the area under the curve
    is used to consolidate all possible thresholds within a certain FPR
    range. Finally, the logarithm is used to bias smaller FPRs. The higher
    the logAUC[0.001, 0.1], the better the performance.

    A perfect classifer gets a logAUC[0.001, 0.1] ) of 1, while a random
    classifer gets a logAUC[0.001, 0.1] ) of around 0.0215 (See [2])

    References:
    [1] Mysinger, M.M. and B.K. Shoichet, Rapid Context-Dependent Ligand
    Desolvation in Molecular Docking. Journal of Chemical Information and
    Modeling, 2010. 50(9): p. 1561-1573.
    [2] Liu, Yunchao, et al. "Interpretable Chirality-Aware Graph Neural
    Network for Quantitative Structure Activity Relationship Modeling in
    Drug Discovery." bioRxiv (2022).
    :param true_y: numpy array of the ground truth. Values are either 0
    (inactive) or 1(active).
    :param predicted_score: numpy array of the predicted score (The
    score does not have to be between 0 and 1)
    :param FPR_range: the range for calculating the logAUC formated in
    (x, y) with x being the lower bound and y being the upper bound
    :return: a numpy array of logAUC of size [1,1]
    """

    # FPR range validity check
    if FPR_range == None:
        raise Exception("FPR range cannot be None")
    lower_bound = FPR_range[0]
    upper_bound = FPR_range[1]
    if lower_bound >= upper_bound:
        raise Exception("FPR upper_bound must be greater than lower_bound")

    fpr, tpr, thresholds = roc_curve(true_y, predicted_score, pos_label=1)

    tpr = np.append(tpr, np.interp([lower_bound, upper_bound], fpr, tpr))
    fpr = np.append(fpr, [lower_bound, upper_bound])

    # Sort both x-, y-coordinates array
    tpr = np.sort(tpr)
    fpr = np.sort(fpr)

    # Get the data points' coordinates. log_fpr is the x coordinate, tpr is the y coordinate.
    log_fpr = np.log10(fpr)
    x = log_fpr
    y = tpr
    lower_bound = np.log10(lower_bound)
    upper_bound = np.log10(upper_bound)

    # Get the index of the lower and upper bounds
    lower_bound_idx = np.where(x == lower_bound)[-1][-1]
    upper_bound_idx = np.where(x == upper_bound)[-1][-1]

    # Create a new array trimmed at the lower and upper bound
    trim_x = x[lower_bound_idx:upper_bound_idx + 1]
    trim_y = y[lower_bound_idx:upper_bound_idx + 1]

    area = auc(trim_x, trim_y) / (upper_bound - lower_bound)
    return area



def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def recall_at_precision_k(y_true, y_pred, threshold=0.9):
    pr, rc, thr = precision_recall_curve(y_true, y_pred)
    if len(np.where(pr >= threshold)[0]) > 0:
        return rc[np.where(pr >= threshold)[0][0]]
    else:
        return 0.0


def precision_at_recall_k(y_true, y_pred, threshold=0.9):
    pr, rc, thr = precision_recall_curve(y_true, y_pred)
    if len(np.where(rc >= threshold)[0]) > 0:
        return pr[np.where(rc >= threshold)[0][-1]]
    else:
        return 0.0


def pcc(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[1, 0]


class Evaluator:
    """Evaluator to evaluate predictions"""
    
    min_metrics = {"mse", "rmse", "mae"}

    def __init__(self):
        self.metric_functions = {
            # Regression metrics
            "mse": mean_squared_error,
            "rmse": rmse,
            "mae": mean_absolute_error,
            "r2": r2_score,
            "pcc": pcc,
            "spearman": stats.spearmanr,

            # Binary classification metrics
            "roc-auc": roc_auc_score,
            "pr-auc": average_precision_score,
            "range_logAUC": range_logAUC,
            "accuracy": accuracy_score,
            "precision": precision_score,
            "recall": recall_score,
            "f1": f1_score,
            "pr@k": precision_at_recall_k,
            "rp@k": recall_at_precision_k,

            # Multi-class classification metrics
            "micro-f1": lambda y_true, y_pred: f1_score(y_true, y_pred, average='micro'),
            "macro-f1": lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro'),
            "micro-precision": lambda y_true, y_pred: precision_score(y_true, y_pred, average='micro'),
            "micro-recall": lambda y_true, y_pred: recall_score(y_true, y_pred, average='micro'),
            "kappa": cohen_kappa_score,
        }

    def __call__(self, metric_name, y_true, y_pred, threshold=None):
        if metric_name in self.metric_functions:
            metric_func = self.metric_functions[metric_name]
            
            if metric_name in ["micro-f1", "macro-f1", "micro-precision", "micro-recall", "kappa"]:
                if isinstance(y_true, list):
                    y_true = np.array(y_true)
                if isinstance(y_pred, list):
                    y_pred = np.array(y_pred)
                if y_true.ndim > 1:
                    y_true = np.argmax(y_true, axis=1)
                if y_pred.ndim > 1:
                    y_pred = np.argmax(y_pred, axis=1)

            if metric_name in ["precision", "recall", "f1", "accuracy"]:
                threshold = 0.5 if threshold is None else threshold
                y_pred = [1 if i > threshold else 0 for i in y_pred]
            if metric_name == "spearman":
                return metric_func(y_true, y_pred)[0]
            return metric_func(y_true, y_pred)
        else:
            raise ValueError(f"Metric {metric_name} is not supported.")

    @staticmethod
    def get_mode(metric_name):
        """Get the mode ('max' or 'min') for the given metric"""
        return 'min' if metric_name in Evaluator.min_metrics else 'max'