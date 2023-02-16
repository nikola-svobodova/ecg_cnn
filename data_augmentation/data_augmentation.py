"""Data augmentation

This file is used to implement the dataset augmentation via undersampling, oversampling and signal transformations.
"""

import wandb
import time
import numpy as np
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from wandb.keras import WandbCallback
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle, class_weight
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.models import load_model
import random
import logging

logging.basicConfig(level=logging.INFO, filename="log.log", filemode="w",
                    format="%(asctime)s - %(levelname)s - %(message)s")

from utils.prepare_dataset import load_data, num_of_types
from basic_model.model import build_model
from data_augmentation.signal_transformation import scale, add_gaussian_noise, add_baseline_wander
from utils.constants import MODELS
from utils.constants import OVERSAMPLING, UNDERSAMPLING, SMOTE_method, SIGNAL_TRANS, GAUSS, SCALING, BASELINE, \
    NONE, CLASS_WEIGHT, \
    baseline_endpoint, scaling_factor, gauss_noise_sigma, phase_shift, amplitude
from utils.model_evaluation import average_folds_results, evaluate_final_metrics, optimize_threshold

trans_type = [SCALING, GAUSS, BASELINE, NONE]


def transform_signal(ecg, type_aug):
    """
    Transforms ECG signal

    :param ecg: ndarray
        original ECG signal
    :param type_aug: str
        type of signal transformation e.g. Scaling
    :return: ndarray
        modified ECG signal
    """
    if type_aug == SCALING:
        _scaling_factor = random.uniform(*scaling_factor)
        modified = np.column_stack([scale(ecg[:, lead], _scaling_factor) for lead in range(12)])
    elif type_aug == GAUSS:
        _gauss_noise_sigma = random.uniform(*gauss_noise_sigma)
        modified = np.column_stack([add_gaussian_noise(ecg[:, lead], mu=0,
                                                       sigma=_gauss_noise_sigma) for lead in range(12)])
    elif type_aug == BASELINE:
        _baseline_endpoint = random.choice(baseline_endpoint)
        _phase_shift = random.uniform(*phase_shift)
        _amplitude = random.uniform(*amplitude)
        modified = np.column_stack([add_baseline_wander(ecg[:, lead],
                                                        amplitude=_amplitude, fun_endpoint=_baseline_endpoint,
                                                        phase_shift=_phase_shift) for lead in range(12)])
    else:
        modified = ecg
    return modified


def data_augmentation(X, y, data_augmentation_factor=1, prob=0.2):
    """
    Applies data augmentation via signal transformations to a given dataset

    :param X: ndarray
        data values
    :param y: ndarray
        data labels
    :param data_augmentation_factor: int
        scales given dataset by this factor; 3 -> increases the dataset three times
    :param prob: float
        percentage increase for nonAF class; 0.2 -> increases nonAF class by 20 %
    :return: tuple(ndarray, ndarray)
        modified dataset
    """
    X_aug = []
    y_aug = []

    X_final = X
    y_final = y
    logging.debug(f"Starting data augmentation via signal transformations with factor: {data_augmentation_factor}")
    for j in range(data_augmentation_factor):
        for i in range(X.shape[0]):
            random_prob = random.uniform(0, 1)
            if y[i] == 0 and random_prob > prob:
                continue

            modify = X[i]
            trans_types = np.random.choice(trans_type, 3)

            for t in trans_types:
                modify = transform_signal(modify, t)

            X_aug.append(modify)
            y_aug.append(y[i])

        X_final = np.concatenate((X_final, np.array(X_aug)), axis=0)
        y_final = np.concatenate((y_final, np.array(y_aug)), axis=0)

        prob = 1
        logging.debug(f"{j} round finished")

    return shuffle(X_final, y_final)


def resample(X, y, method):
    """
    Reshapes dataset and then calls a given re-sampling method

    :param X: ndarray
        data values
    :param y: ndarray
        data labels
    :param method: str
        oversampling x undersampling x SMOTE
    :return: tuple(ndarray, ndarray)
        modified dataset
    """
    X_reshaped = X.reshape(X.shape[0], -1)

    logging.debug(f"Starting {method}")
    if method == OVERSAMPLING:
        oversampler = RandomOverSampler(random_state=47)
        X_reshaped_resampled, y_train = oversampler.fit_resample(X_reshaped, y)
    elif method == UNDERSAMPLING:
        undersampler = RandomUnderSampler(random_state=47)
        X_reshaped_resampled, y_train = undersampler.fit_resample(X_reshaped, y)
    else:
        oversampler = SMOTE(random_state=47)
        X_reshaped_resampled, y_train = oversampler.fit_resample(X_reshaped, y)

    logging.debug(f"{method} is finished")

    X_train = X_reshaped_resampled.reshape(X_reshaped_resampled.shape[0], 4096, 12)

    return shuffle(X_train, y_train)


def train(X_train, y_train, X_val, y_val, run_name, add_class_weights):
    """
    Trains model and logs training results to Weights and Biases (wandb).

    :param X_train: ndarray
        data values
    :param y_train: ndarray
        data labels
    :param X_val: ndarray
        data values
    :param y_val: ndarray
        data labels
    :param run_name: str
       run name
    :param add_class_weights: bool
        add class weights
    :return: str
        best run filepath
    """
    wandb.init(
        name=run_name,
        project="data-augmentation"
    )

    logging.debug("Building model with default configuration")
    model = build_model(kernel_size=32, unit1_filters=16, fc_dropout=0.8, conv_dropout=0.25, l2_regularization=0)

    best_run_name = MODELS + run_name + "_best.hdf5"
    last_run_name = MODELS + run_name + "_last.hdf5"
    callbacks = [
        WandbCallback(
            name=run_name,
            project="data-augmentation"
        ),
        ReduceLROnPlateau(monitor='val_loss',
                          factor=0.1,
                          patience=5),
        ModelCheckpoint(last_run_name),
        ModelCheckpoint(best_run_name, save_best_only=True),
        EarlyStopping(patience=11,
                      min_delta=0.0001)
    ]

    if add_class_weights:
        weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
        weights = {i: weights[i] for i in range(2)}
    else:
        weights = None

    logging.debug("Compiling model")
    model.compile(loss='binary_crossentropy', optimizer=Adam(0.0001), metrics=['accuracy'])

    logging.debug("Training model")
    model.fit(x=X_train, y=y_train, epochs=1, batch_size=32,
              validation_data=(X_val, y_val), class_weight=weights, callbacks=callbacks)
    logging.debug("Finished training model")

    return best_run_name


def evaluate_model(model_filepath, X_test, y_test, X_val, y_val, run_name):
    """
    Evaluates model and logs results to Weights and Biases (wandb).

    :param model_filepath: str
        model filename
    :param X_test: ndarray
        data values
    :param y_test: ndarray
        data labels
    :param X_val: ndarray
        data values
    :param y_val: ndarray
        data labels
    :param run_name: str
        run name
    :return: Metrics
        model evaluation
    """
    logging.debug("Loading model")
    loaded_model = load_model(model_filepath, compile=False)
    loaded_model.compile(loss='binary_crossentropy', optimizer=Adam())

    logging.debug("Optimizing threshold")
    y_prob_ot = loaded_model.predict(X_val)
    optimal_threshold = optimize_threshold(y_true=y_val, y_prob=y_prob_ot, run_name=run_name)

    y_prob = loaded_model.predict(X_test)
    y_prob_2_class = np.hstack([1 - y_prob, y_prob])

    logging.debug("Evaluating model")
    model_evaluation, interp_size = evaluate_final_metrics(
        y_true=y_test,
        y_prob=y_prob,
        optimal_thresh=optimal_threshold,
        run_name=run_name
    )

    logging.info(f"Model evaluation:\nAUPRC: {model_evaluation.auprc}\nF1-score: {model_evaluation.f1}\n"
                 f"accuracy: {model_evaluation.accuracy}\nsensitivity: {model_evaluation.sen}\n"
                 f"specificity: {model_evaluation.spe}\noptimal_threshold: {model_evaluation.optimal_threshold}")

    wandb.run.log({
        'prc-AF': wandb.plot.pr_curve(y_true=y_test, y_probas=y_prob_2_class, interp_size=interp_size,
                                      classes_to_plot=[1]),
        'AUPRC': model_evaluation.auprc,
        'F1-score': model_evaluation.f1,
        'accuracy': model_evaluation.accuracy,
        'sensitivity': model_evaluation.sen,
        'specificity': model_evaluation.spe,
        'optimal_threshold': model_evaluation.optimal_threshold
    })
    wandb.run.finish()

    logging.debug("Finished wandb logging")

    return model_evaluation


def run(method):
    """
    Randomly splits dataset, builds and runs model three times, averages and logs results

    :param method: str
        oversampling x undersampling x SMOTE
    """
    num_runs = 3
    X, y = load_data()
    X_test, y_test = load_data('_test')

    run_results = []

    logging.debug("Starting runs")
    for _ in range(num_runs):
        X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, train_size=0.8888)

        logging.info(f"Train shape {y_train.shape[0]}")
        logging.info(f"Validation shape {y_val.shape[0]}")

        num_of_types(y_train)

        if method is OVERSAMPLING or method is UNDERSAMPLING or method is SMOTE_method:
            X_train, y_train = resample(X=X_train, y=y_train, method=method)
        elif method is SIGNAL_TRANS:
            X_train, y_train = data_augmentation(X=X_train, y=y_train)
        add_class_weight = False

        if method is CLASS_WEIGHT:
            add_class_weight = True

        num_of_types(y_train)

        run_name = f'{method}_{_}_{int(time.time())}'

        best_model_file_name = train(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
                                     run_name=run_name, add_class_weights=add_class_weight)

        model_evaluation = evaluate_model(
            model_filepath=best_model_file_name,
            X_test=X_test,
            y_test=y_test,
            X_val=X_val,
            y_val=y_val,
            run_name=run_name
        )
        run_results.append(model_evaluation)

        logging.debug(f"{_} run is finished")

    logging.debug("All runs are finished")

    run = wandb.init(
        name=f'{method}_{int(time.time())}',
        project="data-augmentation"
    )

    run.log(average_folds_results(run_results))
    logging.debug("Finished wandb logging")

    wandb.run.finish()


if __name__ == "__main__":
    run(SIGNAL_TRANS)
