import tensorflow as tf
import tensorflow.keras.backend as K
from keras.layers import *
from keras.models import *

from src.bubble_detection.model.tcn import TCN


def get_linear_classifier():
    model = Sequential()
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[f2])
    return model


def get_LSTM_classifier(train_set, WIN=30, channels=1, code_size=12, layers=[64,32]):
    model = Sequential()
    norm = tf.keras.layers.Normalization()
    norm.adapt(train_set)
    model.add(norm)
    model.add(LSTM(layers[0], input_shape=(WIN, channels), return_sequences=True))
    model.add(BatchNormalization())
    model.add(LSTM(layers[1]))
    model.add(BatchNormalization())
    model.add(Dense(WIN, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(int(WIN / 2), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(code_size, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[f2])
    model.summary()
    return model


def get_TCN_classifier(WIN, code_size, channels=3, nb_filters=64, kernel_size=4, nb_stacks=2, dilations=None,
                       padding='causal', use_skip_connections=True, dropout_rate=0):
    if dilations is None:
        dilations = [1, 2, 4, 8]

    inputs = Input((WIN, channels))
    xrep = TCN(nb_filters, kernel_size, nb_stacks, dilations, padding,
               use_skip_connections, dropout_rate, return_sequences=True, activation='relu',
               kernel_initializer='random_normal', use_batch_norm=True)(inputs)
    xrep = Flatten()(xrep)
    # xrep = Dense(WIN, activation='relu')(xrep)
    # xrep = Dense(int(WIN / 2), activation='relu')(xrep)
    # xrep = Dense(code_size, activation='relu')(xrep)
    # xrep = Dense(1, activation='sigmoid')(xrep)

    xrep = Dense((WIN), activation='relu')(xrep)
    xrep = BatchNormalization()(xrep)
    xrep = Dense(int(WIN / 2), activation='relu')(xrep)
    xrep = BatchNormalization()(xrep)
    xrep = Dense(code_size, activation='relu')(xrep)
    xrep = Dense(1, activation='sigmoid')(xrep)

    encoder = Model(inputs, xrep)
    encoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=[f2])
    encoder.summary()

    return encoder


def f2(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    agreement = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    total_true_positive = K.sum(K.round(K.clip(y_true, 0, 1)))
    total_pred_positive = K.sum(K.round(K.clip(y_pred, 0, 1)))
    recall = agreement / (total_true_positive + K.epsilon())
    precision = agreement / (total_pred_positive + K.epsilon())
    return (1+2**2)*((precision*recall)/(2**2*precision+recall+K.epsilon()))


def fm(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    agreement = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    total_true_positive = K.sum(K.round(K.clip(y_true, 0, 1)))
    total_pred_positive = K.sum(K.round(K.clip(y_pred, 0, 1)))
    recall = agreement / (total_true_positive + K.epsilon())
    precision = agreement / (total_pred_positive + K.epsilon())
    return tf.math.sqrt(recall * precision)