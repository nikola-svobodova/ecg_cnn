import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import logging

from utils.constants import PICKLE_DATA


def num_of_types(y):
    """
    Returns number of examples for each class in a dataset.

    :param y: nd_array
        data labels
    """
    num_of_sr = np.count_nonzero(y == 0)
    num_of_af = np.count_nonzero(y == 1)

    logging.info(f"No. of SR + oters: {num_of_sr}")
    logging.info(f"No. of AF: {num_of_af}")
    logging.info(f"AF makes: {(num_of_af/len(y)) * 100}% of dataset")


def plot_conf_matrix(cm, run_name):
    """
    Plots confusion matrix

    :param cm: ndarray
        confusion matrix
    :param run_name: str
        run name
    """
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap='Blues')

    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['positive', 'negative'])
    ax.yaxis.set_ticklabels(['positive', 'negative'])

    plt.savefig('/charts/conf_matrix/' + run_name + '.svg')


def plot_pr_curve(precision, recall, auprc, y_test, run_name):
    """
    Plots PR curve.

    :param precision: ndarray
        precision
    :param recall: ndarray
        recall
    :param auprc: ndarray
        area under the PR curve
    :param y_test: ndarray
        true data labels
    :param run_name: str
        run name
    """
    plt.style.use('ggplot')

    fig, ax = plt.subplots()

    no_skill = (np.count_nonzero(y_test == 1) / len(y_test)) * np.ones(len(recall))
    ax.plot(recall, no_skill, color='#2A3132', linewidth=1, linestyle='--', label='No Skill')

    label = 'AF (AUPRC = ' + str(auprc) + ')'
    ax.plot(recall, precision, color='#336B87', label=label)

    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')
    ax.legend()

    plt.savefig('/charts/PR_curve/' + run_name + '.svg')


def plot_f1_threshold_distribution(f1, thresholds, optimal_thr, run_name):
    """
    Plot F1-score distribution

    :param f1: ndarray
        f1-score
    :param thresholds: ndarray
        thresholds
    :param optimal_thr: float
        optimal threshold
    :param run_name: str
        run name
    """
    plt.style.use('ggplot')

    fig, ax = plt.subplots()

    ax.plot(thresholds, f1, color='#A43820')
    ax.vlines(optimal_thr, 0, 1, linestyles="--", color='#2A3132')

    ax.set_ylabel('F1 Score')
    ax.set_xlabel('Threshold')
    ax.legend()

    plt.savefig('/charts/F1_threshold_distribution/' + run_name + '.svg')
