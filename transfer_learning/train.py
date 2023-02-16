"""Transfer learning train

This file is used to test different transfer learning approaches.
"""
import wandb
import numpy as np
from scikeras.wrappers import KerasClassifier
from wandb.keras import WandbCallback
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import precision_recall_curve, auc, make_scorer
from sklearn.utils import shuffle, class_weight
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.models import load_model
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
import logging
import time

logging.basicConfig(level=logging.INFO, filename="log.log", filemode="w",
                    format="%(asctime)s - %(levelname)s - %(message)s")

from utils.help_functions import load_data, num_of_types
from utils.constants import MODELS, FEATURE_EXTRACTION, FINE_TUNING, FINALLY_RETRAIN, SMOTE_method, SIGNAL_TRANS, NONE
from adjust_network import build_model, finally_retrain
from data_augmentation.data_augmentation import data_augmentation, resample
from utils.model_evaluation import optimize_threshold, evaluate_final_metrics, average_folds_results


param_grid_feature_extraction = {
    'classifier__model__optimizer': ['RMSprop', 'Adam', 'SGD'],
    'classifier__model__learning_rate': [0.01, 0.001, 0.0001]
}

param_grid_fine_tuning = {
    'classifier__model__optimizer': ['RMSprop', 'Adam', 'SGD'],
    'classifier__model__learning_rate': [0.01, 0.001, 0.0001],
    'classifier__model__dense_layers': [256, 512, 1024],
    'classifier__model__dropout': [0, 0.25, 0.5]
}

transfer_learning_method = FEATURE_EXTRACTION
weights = False
model_to_retrain = ".../best_model.hdf5"

augmentation_method = SIGNAL_TRANS
data_augmentation_factor = 3


class CustomSmote(BaseEstimator):
    """Custom estimator to include SMOTE in imblearn pipeline"""

    def fit_resample(self, X, y):
        return resample(X, y, SMOTE_method)


class CustomDataAugmentation(BaseEstimator):
    """Custom estimator to include data augmentation via signal transformation in imblearn pipeline"""

    def fit_resample(self, X, y):
        return data_augmentation(X=X, y=y, data_augmentation_factor=data_augmentation_factor, prob=1)


def load_dataset():
    """
    Loads dataset

    :return: tuple(ndarray, ndarray)
        dataset
    """

    # include all data
    X, y = load_data()
    X_test, y_test = load_data('_test')

    X_final = np.concatenate((X, np.array(X_test)), axis=0)
    y_final = np.concatenate((y, np.array(y_test)), axis=0)

    X_final, y_final = shuffle(X_final, y_final)

    return X_final, y_final


def create_model(optimizer, learning_rate):
    """
    Builds model and compiles

    :param optimizer: str
        e.g. Adam
    :param learning_rate: float
        e.g. 0.001
    :return: keras.Sequential
        model
    """
    model_ = build_model(transfer_learning_method)
    logging.info(f'Model with optimizer: {optimizer} and learning rate: {learning_rate} created')

    if optimizer == 'Adam':
        model_.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate), metrics=['accuracy'])
    elif optimizer == 'SGD':
        model_.compile(loss='binary_crossentropy', optimizer=SGD(learning_rate), metrics=['accuracy'])
    else:
        model_.compile(loss='binary_crossentropy', optimizer=RMSprop(learning_rate), metrics=['accuracy'])
    return model_


def custom_auprc(y_true, y_prob):
    """
    Custom metrics for GridSearch

    :param y_true: ndarray
        true labels
    :param y_prob: ndarray
        probabilities
    :return: float
        auc
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    return auc(recall, precision)


def refit(X, y, best_params, run_name):
    """
    Custom refit after GridSearch

    :param X: ndarray
        data values
    :param y: ndarray
        data labels
    :param best_params:
        best parameters based on GridSearch
    :param run_name: str
       run name
    :return: str
        best run model filepath
    """
    wandb.init(
        name=run_name,
        project="transfer-learning"
    )

    logging.info("Split dataset")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, stratify=y)

    logging.info("Resampling train subset")

    # custom refit because of data augmentation/SMOTE application
    if augmentation_method is SMOTE_method:
        X_train, y_train = resample(X_train, y_train, SMOTE_method)
    elif augmentation_method is SIGNAL_TRANS:
        X_train, y_train = data_augmentation(X_train, y_train, data_augmentation_factor, prob=1)

    num_of_types(y_train)

    logging.info(f"Building model (refit) with optimizer: {best_params.get('classifier__model__optimizer')} "
                 f"and learning rate: {best_params.get('classifier__model__learning_rate')}")

    model = create_model(best_params.get('classifier__model__optimizer'),
                         best_params.get('classifier__model__learning_rate'))

    filename_model_best = f'{MODELS}/{run_name}_best.hdf5'
    callbacks = [
        WandbCallback(),
        ReduceLROnPlateau(monitor='val_loss',
                          factor=0.1,
                          patience=5),
        EarlyStopping(patience=10,
                      min_delta=0.0001),
        ModelCheckpoint(filename_model_best, save_best_only=True)
    ]
    weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    weights = {i: weights[i] for i in range(2)}

    logging.info('Refit model')
    model.fit(X_train, y_train, epochs=50, batch_size=32,
              validation_data=(X_val, y_val), callbacks=callbacks, class_weight=weights)

    return filename_model_best


def evaluate_model(best_run_name, X_test, y_test, X_train, y_train, run_name):
    """
    Evaluates model and logs results to Weights and Biases (wandb).

    :param best_run_name: str
        model filename
    :param X_test: ndarray
        data values
    :param y_test: ndarray
        data labels
    :param X_train: ndarray
        data values
    :param y_train: ndarray
        data labels
    :param run_name: str
        run name
    :return: Metrics
        model evaluation
    """
    logging.debug("Loading model")
    loaded_model = load_model(best_run_name, compile=False)
    loaded_model.compile(loss='binary_crossentropy', optimizer=Adam())

    logging.debug("Optimizing threshold")
    y_prob_ot = loaded_model.predict(X_train)
    optimal_threshold = optimize_threshold(y_true=y_train, y_prob=y_prob_ot, run_name=run_name)

    logging.debug("Predicting values")
    y_prob = loaded_model.predict(X_test)

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


def second_phase_fine_tuning(X, y, run_name):
    """
    Pretrains the entire model with a small learning rate

    :param X: ndarray
        data values
    :param y: ndarray
        data labels
    :param run_name: str
        run name
    :return: str
        best run model filepath
    """
    wandb.init(
        name=run_name,
        project="transfer-learning"
    )

    logging.info("Split dataset")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, stratify=y)

    logging.info("Resampling train subset")

    if augmentation_method is SMOTE_method:
        X_train, y_train = resample(X_train, y_train, SMOTE_method)
    elif augmentation_method is SIGNAL_TRANS:
        X_train, y_train = data_augmentation(X=X_train, y=y_train, data_augmentation_factor=3, prob=1)

    num_of_types(y_train)

    logging.info(f"Loading model")
    model = finally_retrain(model_filename=model_to_retrain)

    logging.info(f"Compile model with optimizer: RMSprop and learning rate: 1e-5")
    model.compile(loss='binary_crossentropy', optimizer=RMSprop(1e-5), metrics=['accuracy'])

    filename_model_best = f'{MODELS}/{run_name}_best.hdf5'
    callbacks = [
        WandbCallback(),
        ReduceLROnPlateau(monitor='val_loss',
                          factor=0.1,
                          patience=5),
        EarlyStopping(patience=10,
                      min_delta=0.0001),
        ModelCheckpoint(filename_model_best, save_best_only=True)
    ]
    logging.info('Fit model')
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), callbacks=callbacks)

    return filename_model_best


def cross_validation():
    """
    Nested stratified cross-validation.

    """
    cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=579)
    cv_inner = StratifiedKFold(n_splits=2, shuffle=True, random_state=579)

    X, y = load_dataset()

    run_results = []
    for fold, (train_index, test_index) in enumerate(cv_outer.split(X, y), start=1):
        logging.info(f'NestedCV: {fold} of outer fold {cv_outer.get_n_splits()}')

        run_name = f'{transfer_learning_method}_{augmentation_method}_{fold}_{int(time.time())}'

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if transfer_learning_method is FEATURE_EXTRACTION or transfer_learning_method is FINE_TUNING:
            # create custom scorer AUPRC
            custom_auprc_scorer = make_scorer(custom_auprc, greater_is_better=True, needs_proba=True)

            classifier = KerasClassifier(model=create_model, loss="binary_crossentropy", epochs=5)

            logging.info(f'Creating pipeline.')
            if augmentation_method is SIGNAL_TRANS:
                data_aug = CustomDataAugmentation()
                pipeline = Pipeline([
                    ('sampling', data_aug),
                    ('classifier', classifier)
                ])
            elif augmentation_method is SMOTE_method:
                smot = CustomSmote()
                pipeline = Pipeline([
                    ('sampling', smot),
                    ('classifier', classifier)
                ])
            else:
                pipeline = classifier

            logging.info(f'Starting inner cross-validation.')
            gs = GridSearchCV(estimator=pipeline, param_grid=param_grid_feature_extraction, scoring=custom_auprc_scorer,
                              cv=cv_inner, refit=False, verbose=1)
            gs.fit(X_train, y_train)

            logging.info(f'Finished inner cross-validation.')

            best_params = gs.best_params_
            logging.info(f"Best: {gs.best_score_} AUPRC using {best_params}")

            best_model_filename = refit(X=X_train, y=y_train, best_params=best_params, run_name=run_name)
        elif transfer_learning_method is FINALLY_RETRAIN:
            best_model_filename = second_phase_fine_tuning(X=X_train, y=y_train, run_name=run_name)
        else:
            raise ValueError()

        model_evaluation = evaluate_model(best_run_name=best_model_filename, X_test=X_test, y_test=y_test,
                                          X_train=X_train, y_train=y_train, run_name=run_name)
        run_results.append(model_evaluation)

        logging.debug(f"{fold} run is finished")

    logging.debug("All runs are finished")

    run = wandb.init(
        name=f'{transfer_learning_method}_{augmentation_method}_{int(time.time())}',
        project="transfer-learning"
    )

    run.log(average_folds_results(run_results))
    logging.debug("Finished wandb logging")

    wandb.run.finish()


if __name__ == "__main__":
    cross_validation()
