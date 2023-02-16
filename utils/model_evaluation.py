import logging
import numpy as np
from imblearn.metrics import sensitivity_score, specificity_score
from sklearn.metrics import precision_recall_curve, auc, f1_score, accuracy_score

from utils.plot_results import plot_pr_curve, plot_f1_threshold_distribution, plot_conf_matrix

METHOD = 'charts/example'


class Metrics:
    """
    A class for storing and manipulating model results
    """
    def __init__(self, auprc, accuracy, f1, sen, spe, optimal_threshold):
        self.auprc = auprc
        self.accuracy = accuracy
        self.f1 = f1
        self.sen = sen
        self.spe = spe
        self.optimal_threshold = optimal_threshold


def optimize_threshold(y_true, y_prob, run_name):
    """
    Takes a list of predicted probabilities and their corresponding true labels,
    and calculates the optimal threshold for classification. The optimal threshold is the
    value that maximizes the F1 score.

    :param y_true: ndarray
        true values
    :param y_prob: ndarray
        predicted probabilities
    :param run_name: str
       run name
    :return: float
        optimal threshold
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # pr_auc = auc(recall, precision)
    # plot_pr_curve(precision, recall, round(pr_auc, 2), y_true, f"{METHOD}_train_{run_name}")

    # get F1-score for each threshold
    num = 2 * np.multiply(precision, recall)
    denom = np.add(precision, recall)
    f1 = np.divide(num, denom, out=np.zeros_like(num), where=denom != 0)[:-1]

    optimal_index = np.argmax(f1)
    optimal_threshold = thresholds[optimal_index]
    logging.info(f"optimal_threshold: {optimal_threshold}")

    # plot_f1_threshold_distribution(f1, thresholds, optimal_threshold, f"{METHOD}_train_{run_name}")
    return optimal_threshold


def evaluate_final_metrics(y_true, y_prob, optimal_thresh, run_name):
    """
    Takes the true labels and predicted labels as input, and calculates a set of
    common classification metrics to evaluate the performance of the model. The metrics that are
    calculated include accuracy, F1 score, specificity, sensitivity and the area under the PR curve.

    :param y_true: ndarray
        true values
    :param y_prob: ndarray
        predicted probabilities
    :param optimal_thresh: float
        optimal threshold
    :param run_name: str
       run name
    :return: Metrics
        model evaluation
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)

    y_pred = (y_prob > optimal_thresh).astype(int)

    # plot_conf_matrix(cm=confusion_matrix(y_true=y_true, y_pred=y_pred), run_name=f"{METHOD}_{run_name}")

    f1 = f1_score(y_true=y_true, y_pred=y_pred)
    sen = sensitivity_score(y_true=y_true, y_pred=y_pred)
    spe = specificity_score(y_true=y_true, y_pred=y_pred)
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)

    return Metrics(
        auprc=pr_auc,
        accuracy=accuracy,
        f1=f1,
        sen=sen,
        spe=spe,
        optimal_threshold=optimal_thresh
    ), len(recall)


def get_mean_and_std(values):
    """
    Returns mean and std

    :param values:
    :return: tuple[float, float]
        mean, std
    """
    return np.mean(values), np.std(values)


def average_folds_results(scores):
    """
    Takes a list of metrics and calculates the average of each

    :param scores:
    :return: dict
        average metrics
    """
    auprc_avg, auprc_std = get_mean_and_std([metric.auprc for metric in scores])
    acc_avg, acc_std = get_mean_and_std([metric.accuracy for metric in scores])
    f1_avg, f1_std = get_mean_and_std([metric.f1 for metric in scores])
    sen_avg, sen_std = get_mean_and_std([metric.sen for metric in scores])
    spe_avg, spe_std = get_mean_and_std([metric.spe for metric in scores])

    return {
        'average_AUPRC': auprc_avg,
        'average_F1-score': f1_avg,
        'average_accuracy': acc_avg,
        'average_sensitivity': sen_avg,
        'average_specificity': spe_avg,
    }
