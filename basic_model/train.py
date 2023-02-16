"""Basic model hyperparameters optimalization / train / evaluate

This file is used to optimize basic model hyperparameters
"""
import time
import yaml
import wandb
import numpy as np
from wandb.keras import WandbCallback
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve, auc
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
import logging

from utils.model_evaluation import average_folds_results

logging.basicConfig(level=logging.INFO, filename="log.log", filemode="w",
                    format="%(asctime)s - %(levelname)s - %(message)s")

from basic_model.model import build_model
from utils.prepare_dataset import load_data


def train(X_train, y_train, X_val, y_val):
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
    """
    run = wandb.init(
        config=wandb.config,
        project="basic-model-tuning"
    )

    model = build_model(
        kernel_size=wandb.config.kernel_size,
        unit1_filters=wandb.config.unit1_filters,
        conv_dropout=wandb.config.conv_dropout,
        fc_dropout=wandb.config.fc_dropout,
        l2_regularization=wandb.config.l2_regularization
    )

    callbacks = [
        WandbCallback(),
        ReduceLROnPlateau(monitor='val_loss',
                          factor=0.1,
                          patience=5)
    ]

    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=wandb.config.lr), metrics=['accuracy'])

    model.fit(x=X_train, y=y_train, epochs=20, batch_size=wandb.config.batch_size,
              validation_data=(X_val, y_val), callbacks=callbacks)

    y_prob = model.predict(X_val)
    y_prob_2_class = np.hstack([1 - y_prob, y_prob])

    precision, recall, thresholds = precision_recall_curve(y_val, y_prob)
    pr_auc = auc(recall, precision)

    # Log the precision-recall curve to wandb
    run.log({
        'prc-AF': wandb.plot.pr_curve(y_true=y_val, y_probas=y_prob_2_class, interp_size=len(recall),
                                      classes_to_plot=[1]),
        'AUPRC': pr_auc,
    })

    wandb.run.finish()
    logging.debug("Finished wandb logging")


def run():
    """
    Randomly splits dataset, builds and runs model three times, averages and logs results

    """
    n_folds = 3
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=375)

    X, y = load_data()

    run_results = []

    logging.debug("Starting runs")
    for train_index, val_index in cv.split(X, y):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        logging.info(f"Train shape {y_train.shape[0]}")
        logging.info(f"Validation shape {y_val.shape[0]}")

        train(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)

    logging.debug("All runs are finished")

    run = wandb.init(
        name=f'average_{int(time.time())}',
        project="data-augmentation"
    )

    run.log(average_folds_results(run_results))
    logging.debug("Finished wandb logging")

    wandb.run.finish()


def sweep():
    """
    Initializes sweep and starts agent
    """
    with open('sweep.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    sweep_id = wandb.sweep(sweep=config, project="basic-model-tuning")
    wandb.agent(sweep_id=sweep_id, function=run)


if __name__ == "__main__":
    sweep()
    wandb.finish()
