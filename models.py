import kapre
import numpy as np
import tensorflow as tf
from kapre.composed import get_melspectrogram_layer
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Cropping2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    LayerNormalization,
    MaxPooling2D,
    UpSampling2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


def classifier(N_CLASSES=4, SR=16000, DT=10.0, N_CHANNELS=8):
    """builds and compile the classifier model

    Args:
        N_CLASSES (int, optional): number of classes. Defaults to 4.
        SR (int, optional): sampling rate of the audio. Defaults to 16000.
        DT (float, optional): time of each audio file. Defaults to 10.0.
        N_CHANNELS (int, optional): number of audio channels. Defaults to 8.

    Returns:
        model: compiled classifier model
    """
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
    model = Model(inputs=i.input, outputs=o, name="classifier")
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def anomaly_detector(ID, N_CHANNELS=8, SR=16000, DT=10.0, N_MELS=128, HOP_LENGTH=512):
    """builds and compile the anomaly detector model

    Args:
        ID (string): model identification
        N_CHANNELS (int, optional): number of audio channels. Defaults to 8.
        SR (int, optional): audio sampling rate. Defaults to 16000.
        DT (float, optional): time of each audio file. Defaults to 10.0.
        N_MELS (int, optional): number of melspectrogram. Defaults to 128.
        HOP_LENGTH (int, optional): hop length. Defaults to 512.

    Returns:
        model: compiled anomaly detector model
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
        8,
        kernel_size=(3, 3),
        activation="relu",
        padding="same",
        name="enc_conv2d_relu_1",
    )(i)
    x = Conv2D(
        8,
        kernel_size=(3, 3),
        activation="relu",
        padding="same",
        name="enc_conv2d_relu_1a",
    )(x)
    x = MaxPooling2D(pool_size=(2, 2), padding="same", name="enc_maxpool2d_1")(x)
    x = Conv2D(
        4,
        kernel_size=(3, 3),
        activation="relu",
        padding="same",
        name="enc_conv2d_relu_2",
    )(x)
    x = Conv2D(
        4,
        kernel_size=(3, 3),
        activation="relu",
        padding="same",
        name="enc_conv2d_relu_2a",
    )(x)
    x = MaxPooling2D(pool_size=(2, 2), padding="same", name="enc_maxpool2d_2")(x)
    x = Conv2D(
        2,
        kernel_size=(3, 3),
        activation="relu",
        padding="same",
        name="enc_conv2d_relu_3",
    )(x)
    x = Conv2D(
        2,
        kernel_size=(3, 3),
        activation="relu",
        padding="same",
        name="enc_conv2d_relu_3a",
    )(x)
    x = MaxPooling2D(pool_size=(2, 2), padding="same", name="enc_maxpool2d_3")(x)

    # get the dimension before flatten
    # to be used at the decoder for reshaping
    p, q, r = x.shape[1], x.shape[2], x.shape[3]

    x = Flatten(name="enc_flatten_1")(x)
    x = Dense(600, activation="relu", kernel_regularizer="l1_l2", name="enc_dense_1")(x)
    x = Dense(100, activation="relu", kernel_regularizer="l1_l2", name="enc_dense_2")(x)
    e = Dense(10, activation="relu", kernel_regularizer="l1_l2", name="enc_dense_3")(x)
    ##### ENCODER ENDS #####

    ##### DECODER STARTS #####
    x = Dense(100, activation="relu", name="dec_dense_1")(e)
    x = Dense(600, activation="relu", kernel_regularizer="l1_l2", name="dec_dense_2")(x)
    x = Dense(
        p * q * r, activation="relu", kernel_regularizer="l1_l2", name="dec_dense_3"
    )(x)
    x = tf.reshape(x, [-1, p, q, r], name="dec_reshape_1")

    x = Conv2D(
        2,
        kernel_size=(3, 3),
        activation="relu",
        padding="same",
        name="dec_conv2d_relu_1",
    )(x)
    x = Conv2D(
        2,
        kernel_size=(3, 3),
        activation="relu",
        padding="same",
        name="dec_conv2d_relu_1a",
    )(x)
    x = UpSampling2D(size=(2, 2), name="dec_upsampling2d_1")(x)
    x = Conv2D(
        4,
        kernel_size=(3, 3),
        activation="relu",
        padding="same",
        name="dec_conv2d_relu_2",
    )(x)
    x = Conv2D(
        4,
        kernel_size=(3, 3),
        activation="relu",
        padding="same",
        name="dec_conv2d_relu_2a",
    )(x)
    x = UpSampling2D(size=(2, 2), name="dec_upsampling2d_2")(x)
    x = Conv2D(
        8,
        kernel_size=(3, 3),
        activation="relu",
        padding="same",
        name="dec_conv2d_relu_3",
    )(x)
    x = Conv2D(
        8,
        kernel_size=(3, 3),
        activation="relu",
        padding="same",
        name="dec_conv2d_relu_3a",
    )(x)
    x = UpSampling2D(size=(2, 2), name="dec_upsampling2d_3")(x)
    # crop the output
    # reference:
    # https://stats.stackexchange.com/questions/376464/convolutional-autoencoder-on-an-odd-size-image
    d = Cropping2D(cropping=((0, 0), (3, 3)), data_format=None, name="output")(x)
    ##### DECODER ENDS #####

    model = Model(inputs=i, outputs=d, name=ID)

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.MeanSquaredError(),
    )
    return model
