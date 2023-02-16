"""Model

Build model with defined hyperparameters
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
from keras import regularizers


def build_model(kernel_size=16, unit1_filters=32, fc_dropout=0.5, conv_dropout=0.25, l2_regularization=0):
    """
    Builds model

    :param kernel_size: int
        kernel size
    :param unit1_filters: int
        number of filters in the first convolutional layer
    :param fc_dropout: float
        dropout rate between fully connected layers
    :param conv_dropout: float
        dropout rate between convolutional layers
    :param l2_regularization: float
        l2 regularization parameter
    :return: keras.Sequential
        model
    """
    model = Sequential()

    model.add(Conv1D(unit1_filters, kernel_size, padding='same', activation='relu',
                     kernel_regularizer=regularizers.l2(l2_regularization), input_shape=(4096, 12)))
    model.add(Conv1D(unit1_filters, kernel_size, activation='relu',
                     kernel_regularizer=regularizers.l2(l2_regularization)))
    model.add(MaxPooling1D(2))
    model.add(Dropout(conv_dropout))

    model.add(Conv1D(2 * unit1_filters, kernel_size, padding='same', activation='relu',
                     kernel_regularizer=regularizers.l2(l2_regularization)))
    model.add(Conv1D(2 * unit1_filters, kernel_size, activation='relu',
                     kernel_regularizer=regularizers.l2(l2_regularization)))
    model.add(MaxPooling1D(2))
    model.add(Dropout(conv_dropout))

    model.add(Conv1D(4 * unit1_filters, kernel_size, padding='same', activation='relu',
                     kernel_regularizer=regularizers.l2(l2_regularization)))
    model.add(Conv1D(4 * unit1_filters, kernel_size, activation='relu',
                     kernel_regularizer=regularizers.l2(l2_regularization)))
    model.add(MaxPooling1D(2))
    model.add(Dropout(conv_dropout))

    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(l2_regularization)))
    model.add(Dropout(fc_dropout))

    model.add(Dense(1, activation="sigmoid"))

    return model


if __name__ == "__main__":
    model = build_model()
    model.summary()
