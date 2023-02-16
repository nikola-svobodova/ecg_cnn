"""Adjust network

File to adjust network
"""
from keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, MaxPooling1D, BatchNormalization
import tensorflow as tf

from utils.constants import FEATURE_EXTRACTION, FINE_TUNING, BASE_MODEL, TRAIN_FROM_SCRATCH, \
    FULLY_RETRAIN, FINALLY_RETRAIN


def load_base_model():
    model_hdf5 = f'.../model.hdf5'
    base_model = load_model(model_hdf5, compile=False)
    return base_model


def train_from_scratch():
    model_with_weights = load_base_model()

    base_model = tf.keras.models.Model.from_config(model_with_weights.get_config())

    base_model_pop_layers = tf.keras.Model(base_model.input, base_model.layers[48].output)
    # check base model is fully unfreeze
    assert (49 == len([i for i in base_model_pop_layers.layers if i.trainable is True]))

    model = Sequential()
    model.add(base_model_pop_layers)
    model.add(Dense(1, activation='sigmoid'))

    return model


def feature_extraction():
    base_model = load_base_model()

    for layer in base_model.layers:
        layer.trainable = False

    base_model_pop_layers = tf.keras.Model(base_model.input, base_model.layers[48].output)

    # check base model is fully freeze
    assert (49 == len([i for i in base_model_pop_layers.layers if i.trainable is False]))

    model = Sequential()
    model.add(base_model_pop_layers)
    model.add(Dense(1, activation="sigmoid"))

    return model


def fully_retrain():
    base_model = load_base_model()

    base_model_pop_layers = tf.keras.Model(base_model.input, base_model.layers[48].output)

    # check base model is fully freeze
    assert (49 == len([i for i in base_model_pop_layers.layers if i.trainable is True]))

    model = Sequential()
    model.add(base_model_pop_layers)
    model.add(Dense(1, activation="sigmoid"))

    return model


def fine_tune(dropout=0, first_dense_layer=512, second_dense_layer=0, freeze_layers=48):
    base_model = load_base_model()

    for layer in base_model.layers:
        layer.trainable = False

    base_model_pop_layers = tf.keras.Model(base_model.input, base_model.layers[freeze_layers].output)

    # check base model is freeze
    assert (freeze_layers + 1 == len([i for i in base_model_pop_layers.layers if i.trainable is False]))

    model = Sequential()
    model.add(base_model_pop_layers)
    model.add(Dense(first_dense_layer, activation='relu'))
    model.add(Dropout(dropout))
    if second_dense_layer != 0:
        model.add(Dense(second_dense_layer, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model


def finally_retrain(model_filename):
    model = load_model(model_filename, compile=False)

    for layer in model.layers:
        layer.trainable = True

    return model


def build_model(
        tl_type,
        model_filename='',
        dropout=0,
        first_dense_layer=512,
        second_dense_layer=0,
        freeze_layers=48

):
    if tl_type == BASE_MODEL:
        model = load_base_model()
    elif tl_type == FEATURE_EXTRACTION:
        model = feature_extraction()
    elif tl_type == TRAIN_FROM_SCRATCH:
        model = train_from_scratch()
    elif tl_type == FINE_TUNING:
        model = fine_tune(dropout, first_dense_layer, second_dense_layer, freeze_layers)
    elif tl_type == FULLY_RETRAIN:
        model = fully_retrain()
    elif tl_type == FINALLY_RETRAIN:
        model = finally_retrain(model_filename)
    else:
        raise ValueError(f'unknown transfer learning method: {tl_type}')

    return model
