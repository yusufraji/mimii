import numpy as np
import kapre
from kapre.composed import get_melspectrogram_layer

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, TimeDistributed, LayerNormalization, BatchNormalization, ReLU, Dropout, GlobalAveragePooling2D, Softmax, UpSampling2D, Cropping2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, TensorBoard


def MyConv2D(N_CLASSES=4, SR=16000, DT=10.0, N_CHANNELS=8):
    input_shape = (int(SR*DT), N_CHANNELS)
    i = get_melspectrogram_layer(input_shape=input_shape,
                                 n_mels=128,
                                 pad_end=True,
                                 n_fft=512,
                                 win_length=400,
                                 hop_length=160,
                                 sample_rate=SR,
                                 return_decibel=True,
                                 input_data_format='channels_last',
                                 output_data_format='channels_last')
    x = LayerNormalization(axis=2, name='batch_norm')(i.output)
    x = Conv2D(8, kernel_size=(7,7), activation='tanh', padding='same', name='conv2d_tanh')(x)
    x = MaxPooling2D(pool_size=(2,2), padding='same', name='max_pool_2d_1')(x)
    x = Conv2D(16, kernel_size=(5,5), activation='relu', padding='same', name='conv2d_relu_1')(x)
    x = MaxPooling2D(pool_size=(2,2), padding='same', name='max_pool_2d_2')(x)
    x = Conv2D(16, kernel_size=(3,3), activation='relu', padding='same', name='conv2d_relu_2')(x)
    x = MaxPooling2D(pool_size=(2,2), padding='same', name='max_pool_2d_3')(x)
    x = Conv2D(32, kernel_size=(3,3), activation='relu', padding='same', name='conv2d_relu_3')(x)
    x = MaxPooling2D(pool_size=(2,2), padding='same', name='max_pool_2d_4')(x)
    x = Conv2D(32, kernel_size=(3,3), activation='relu', padding='same', name='conv2d_relu_4')(x)
    x = Conv2D(64, kernel_size=(3,3), activation='relu', padding='same', name='conv2d_relu_5')(x)
    x = MaxPooling2D(pool_size=(2,2), padding='same', name='max_pool_2d_5')(x)
    x = Conv2D(64, kernel_size=(3,3), activation='relu', padding='same', name='conv2d_relu_6')(x)
    x = Flatten(name='flatten')(x)
    x = Dropout(rate=0.2, name='dropout')(x)
    x = Dense(64, activation='relu', activity_regularizer=l2(0.001), name='dense')(x)
    o = Dense(N_CLASSES, activation='softmax', name='softmax')(x)
    model = Model(inputs=i.input, outputs=o, name='2d_convolution')
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
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
    n_frames = int(np.floor((SR*DT)/HOP_LENGTH) + 1)
    # n_frames must be even for autoencoder symmetry
    # round n_frames up to the nearest even value!
    n_frames = int(np.ceil(n_frames/2.) * 2)
    input_shape = (N_MELS, n_frames, N_CHANNELS)
    i = Input(shape=input_shape, name='input')
    x = Conv2D(8, kernel_size=(7,7), activation='tanh', padding='same', name='conv2d_tanh_1')(i)
    x = MaxPooling2D(pool_size=(2,2), padding='same', name='max_pool_2d_1')(x)
    x = Conv2D(16, kernel_size=(5,5), activation='relu', padding='same', name='conv2d_relu_1')(x)
    x = MaxPooling2D(pool_size=(2,2), padding='same', name='max_pool_2d_2')(x)
    x = Conv2D(32, kernel_size=(3,3), activation='relu', padding='same', name='conv2d_relu_2')(x)
    x = MaxPooling2D(pool_size=(2,2), padding='same', name='max_pool_2d_3')(x)
    x = Conv2D(32, kernel_size=(3,3), activation='relu', padding='same', name='conv2d_relu_3')(x)
    x = MaxPooling2D(pool_size=(2,2), padding='same', name='max_pool_2d_4')(x)
    x = Conv2D(32, kernel_size=(3,3), activation='relu', padding='same', name='conv2d_relu_4')(x)
    e = MaxPooling2D(pool_size=(2,2), padding='same', name='max_pool_2d_5')(x)
    
    x = Conv2D(32, kernel_size=(3,3), activation='relu', padding='same', name='conv2d_relu_5')(e)
    x = UpSampling2D(size=(2,2), name='up_2d_1')(x)
    x = Conv2D(16, kernel_size=(5,5), activation='relu', padding='same', name='conv2d_relu_6')(x)
    x = UpSampling2D(size=(2,2), name='up_2d_2')(x)
    x = Conv2D(16, kernel_size=(5,5), activation='relu', padding='same', name='conv2d_relu_7')(x)
    x = UpSampling2D(size=(2,2), name='up_2d_3')(x)
    x = Conv2D(16, kernel_size=(5,5), activation='relu', padding='same', name='conv2d_relu_8')(x)
    x = UpSampling2D(size=(2,2), name='up_2d_4')(x)
    x = Conv2D(8, kernel_size=(7,7), activation='tanh', padding='same', name='conv2d_tanh_2')(x)
    x = UpSampling2D(size=(2,2), name='up_2d_5')(x)
    # crop the output
    # reference:
    # https://stats.stackexchange.com/questions/376464/convolutional-autoencoder-on-an-odd-size-image
    d = Cropping2D(cropping=((0, 0), (3, 3)), data_format=None)(x)

    model = Model(inputs=i, outputs=d, name='2d_convolution_autoencoder')
    model.compile(optimizer='adam',
                  loss='mean_squared_error', 
                  metrics=['accuracy'])
    return model

def MyConv1DAE(parameter_list):
    """
    docstring
    """
    pass