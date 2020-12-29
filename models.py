import kapre
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from kapre.composed import get_melspectrogram_layer
from tensorflow.keras.callbacks import (
    EarlyStopping,
    LearningRateScheduler,
    ModelCheckpoint,
    TensorBoard,
)
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Cropping2D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    Input,
    LayerNormalization,
    MaxPooling2D,
    ReLU,
    Softmax,
    TimeDistributed,
    UpSampling2D,
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model, to_categorical


def MyConv2D(N_CLASSES=4, SR=16000, DT=10.0, N_CHANNELS=8):
    input_shape = (int(SR * DT), N_CHANNELS)
    i = get_melspectrogram_layer(
        input_shape=input_shape,
        n_mels=128,
        pad_end=True,
        n_fft=512,
        win_length=400,
        hop_length=160,
        sample_rate=SR,
        return_decibel=True,
        input_data_format="channels_last",
        output_data_format="channels_last",
    )
    x = LayerNormalization(axis=2, name="batch_norm")(i.output)
    x = Conv2D(
        8, kernel_size=(7, 7), activation="tanh", padding="same", name="conv2d_tanh"
    )(x)
    x = MaxPooling2D(pool_size=(2, 2), padding="same", name="max_pool_2d_1")(x)
    x = Conv2D(
        16, kernel_size=(5, 5), activation="relu", padding="same", name="conv2d_relu_1"
    )(x)
    x = MaxPooling2D(pool_size=(2, 2), padding="same", name="max_pool_2d_2")(x)
    x = Conv2D(
        16, kernel_size=(3, 3), activation="relu", padding="same", name="conv2d_relu_2"
    )(x)
    x = MaxPooling2D(pool_size=(2, 2), padding="same", name="max_pool_2d_3")(x)
    x = Conv2D(
        32, kernel_size=(3, 3), activation="relu", padding="same", name="conv2d_relu_3"
    )(x)
    x = MaxPooling2D(pool_size=(2, 2), padding="same", name="max_pool_2d_4")(x)
    x = Conv2D(
        32, kernel_size=(3, 3), activation="relu", padding="same", name="conv2d_relu_4"
    )(x)
    x = Conv2D(
        64, kernel_size=(3, 3), activation="relu", padding="same", name="conv2d_relu_5"
    )(x)
    x = MaxPooling2D(pool_size=(2, 2), padding="same", name="max_pool_2d_5")(x)
    x = Conv2D(
        64, kernel_size=(3, 3), activation="relu", padding="same", name="conv2d_relu_6"
    )(x)
    x = Flatten(name="flatten")(x)
    x = Dropout(rate=0.2, name="dropout")(x)
    x = Dense(64, activation="relu", activity_regularizer=l2(0.001), name="dense")(x)
    o = Dense(N_CLASSES, activation="softmax", name="softmax")(x)
    model = Model(inputs=i.input, outputs=o, name="2d_convolution")
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def MyLSTM(parameter_list):
    """
    docstring
    """
    pass


def MyConv1D(parameter_list):
    """
    docstring
    """
    pass


def MyConv2DAE(N_CHANNELS=8, SR=16000, DT=10.0, N_MELS=128, HOP_LENGTH=512):
    """
    docstring
    """
    # reference:
    # https://stackoverflow.com/questions/51241499/parameters-to-control-the-size-of-a-spectrogram
    n_frames = int(np.floor((SR * DT) / HOP_LENGTH) + 1)
    # n_frames must be even for autoencoder symmetry
    # round n_frames up to the nearest even value!
    n_frames = int(np.ceil(n_frames / 2.0) * 2)
    input_shape = (N_MELS, n_frames, N_CHANNELS)
    ##### ENCODER STARTS #####
    i = Input(shape=input_shape, name="input")
    x = Conv2D(
        8, kernel_size=(7, 7), activation="tanh", padding="same", name="conv2d_tanh_1"
    )(i)
    x = BatchNormalization(name="batch_norm_1")(x)
    x = MaxPooling2D(pool_size=(2, 2), padding="same", name="max_pool_2d_1")(x)
    x = Conv2D(
        16, kernel_size=(5, 5), activation="relu", padding="same", name="conv2d_relu_1"
    )(x)
    x = BatchNormalization(name="batch_norm_2")(x)
    x = MaxPooling2D(pool_size=(2, 2), padding="same", name="max_pool_2d_2")(x)
    x = Conv2D(
        32, kernel_size=(3, 3), activation="relu", padding="same", name="conv2d_relu_2"
    )(x)
    x = BatchNormalization(name="batch_norm_3")(x)
    x = MaxPooling2D(pool_size=(2, 2), padding="same", name="max_pool_2d_3")(x)
    x = Conv2D(
        32, kernel_size=(3, 3), activation="relu", padding="same", name="conv2d_relu_3"
    )(x)
    x = BatchNormalization(name="batch_norm_4")(x)
    x = MaxPooling2D(pool_size=(2, 2), padding="same", name="max_pool_2d_4")(x)
    x = Conv2D(
        32, kernel_size=(3, 3), activation="relu", padding="same", name="conv2d_relu_4"
    )(x)
    x = BatchNormalization(name="batch_norm_5")(x)
    x = MaxPooling2D(pool_size=(2, 2), padding="same", name="max_pool_2d_5")(x)

    # get the dimension before flatten
    # to be used at the decoder for reshaping
    p, q, r = x.shape[1], x.shape[2], x.shape[3]

    x = Flatten(name="flatten_1")(x)
    x = BatchNormalization(name="batch_norm_6")(x)
    x = Dense(600, activation="relu", name="dense_1")(x)
    e = BatchNormalization(name="batch_norm_7")(x)
    ##### ENCODER ENDS #####

    ##### DECODER STARTS #####
    x = Dense(p*q*r, activation="relu", name="dense_2")(e)
    x = BatchNormalization(name="batch_norm_8")(x)
    x = tf.reshape(x, [-1, p, q, r], name="reshape_1")

    x = Conv2D(
        32, kernel_size=(3, 3), activation="relu", padding="same", name="conv2d_relu_5"
    )(x)
    x = BatchNormalization(name="batch_norm_9")(x)
    x = UpSampling2D(size=(2, 2), name="up_2d_1")(x)
    x = Conv2D(
        16, kernel_size=(5, 5), activation="relu", padding="same", name="conv2d_relu_6"
    )(x)
    x = BatchNormalization(name="batch_norm_10")(x)
    x = UpSampling2D(size=(2, 2), name="up_2d_2")(x)
    x = Conv2D(
        16, kernel_size=(5, 5), activation="relu", padding="same", name="conv2d_relu_7"
    )(x)
    x = BatchNormalization(name="batch_norm_11")(x)
    x = UpSampling2D(size=(2, 2), name="up_2d_3")(x)
    x = Conv2D(
        16, kernel_size=(5, 5), activation="relu", padding="same", name="conv2d_relu_8"
    )(x)
    x = BatchNormalization(name="batch_norm_12")(x)
    x = UpSampling2D(size=(2, 2), name="up_2d_4")(x)
    x = Conv2D(
        8, kernel_size=(7, 7), activation="tanh", padding="same", name="conv2d_tanh_2"
    )(x)
    x = BatchNormalization(name="batch_norm_13")(x)
    x = UpSampling2D(size=(2, 2), name="up_2d_5")(x)
    # crop the output
    # reference:
    # https://stats.stackexchange.com/questions/376464/convolutional-autoencoder-on-an-odd-size-image
    d = Cropping2D(cropping=((0, 0), (3, 3)), data_format=None)(x)
    ##### DECODER ENDS #####

    model = Model(inputs=i, outputs=d, name="2d_convolution_autoencoder")
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])
    return model


def MyConv1DAE(parameter_list):
    """
    docstring
    """
    pass
