import time
from datetime import datetime
from pathlib import Path
import argparse
from scipy.io import wavfile
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import tensorflow as tf
tf.config.experimental.list_physical_devices('GPU')
import tensorflow_addons as tfa
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, TensorBoard
from numba import cuda 

from models import MyConv2D
from utils import fetch_dataset, save_fig, model_metrics, plot_history, fetch_dataset


import yaml


# data generator
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, wav_paths, labels, sr, dt, n_classes, batch_size=32, n_channels=1, shuffle=True):
        self.wav_paths = wav_paths
        self.labels = labels
        self.sr = sr
        self.dt = dt
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.shuffle = True
        self.n_channels = n_channels
        self.on_epoch_end()


    def __len__(self):
        return int(np.floor(len(self.wav_paths) / self.batch_size))


    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        wav_paths = [self.wav_paths[k] for k in indexes]

        labels = [self.labels[k] for k in indexes]
        
        # generate a batch of time data
        X = np.empty((self.batch_size, int(self.sr*self.dt), self.n_channels), dtype=np.float32)
        Y = np.empty((self.batch_size, self.n_classes), dtype=np.float32)

        for i, (path, label) in enumerate(zip(wav_paths, labels)):
            rate, wav = wavfile.read(path)
            
            X[i,] = wav[:,:self.n_channels]
            Y[i,] = to_categorical(label, num_classes=self.n_classes)

        return X, Y


    def on_epoch_end(self):
        self.indexes = np.arange(len(self.wav_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)


# create train test and valid set
def train_test_valid(data, n_classes=4, test_size=0.2, valid_size=0.1):

    # integer encoder
    le = LabelEncoder()
    if n_classes == 4:
        integer_encoded = le.fit_transform(data['machine_type'])
    else:
        integer_encoded = le.fit_transform(data["machine_type_id"])
    # Split train, test, and valid set
    X_train_full, X_test, y_train_full, y_test = train_test_split(data['normal'], integer_encoded, test_size=test_size, shuffle=True, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=valid_size, shuffle=True, random_state=42)

    train_df = pd.DataFrame()
    train_df['X_train'] = X_train
    train_df['y_train'] = le.inverse_transform(y_train)

    test_df = pd.DataFrame()
    test_df['X_test'] = X_test
    test_df['y_test'] = le.inverse_transform(y_test)

    return np.asarray(X_train), np.asarray(X_valid), np.asarray(X_test), y_train, y_valid, y_test, train_df, test_df

def train(args):
    """
    docstring
    """
    cur_dir = Path.cwd()
    device = cuda.get_current_device()
    # load config yaml
    with open("config.yaml") as stream:
        config = yaml.safe_load(stream)

    models = {'myconv2d' : MyConv2D(N_CLASSES=config["feature"]["n_classes"], SR=config["feature"]["sr"], DT=config["feature"]["dt"], N_CHANNELS=config["feature"]["n_channels"])}
    assert args.model in models.keys(), f'{args.model} is unavailable.'
    assert config["feature"]["n_classes"] in [4, 16], f'n_classes({config["feature"]["n_classes"]}) must either be 4 or 16'

    # create directories
    Path.mkdir(cur_dir / config["results_dir"], exist_ok=True)
    Path.mkdir(cur_dir / config["log_dir"], exist_ok=True)

    # setup result
    result_file = "{result}/{file}".format(result=config["results_dir"], file=config["result_file"])
    results = {}

    # fetch dataset
    if 'dataset_df.csv' in [x.name for x in (cur_dir / config["dataset_dir"]).iterdir()]:
        data = pd.read_csv(cur_dir / config["dataset_dir"] / 'dataset_df.csv')
    else:
        data = fetch_dataset()

    # train test valid split
    X_train, X_valid, X_test, y_train, y_valid, y_test, train_df, test_df = train_test_valid(data, n_classes=config["feature"]["n_classes"], test_size=config["fit"]["test_size"], valid_size=config["fit"]["valid_size"])
    train_size = len(X_train)
    valid_size = len(X_valid)

    train_df.to_csv(cur_dir / config["dataset_dir"] / 'train_df.csv', index=False)
    test_df.to_csv(cur_dir / config["dataset_dir"] / 'test_df.csv', index=False)

    # dataset generator
    train_gen = DataGenerator(X_train, y_train, config["feature"]["sr"], config["feature"]["dt"],
                        n_classes=config["feature"]["n_classes"], batch_size=config["fit"]["batch_size"], n_channels=config["feature"]["n_channels"])

    valid_gen = DataGenerator(X_valid, y_valid, config["feature"]["sr"], config["feature"]["dt"],
                        n_classes=config["feature"]["n_classes"], batch_size=config["fit"]["batch_size"], n_channels=config["feature"]["n_channels"])

    test_gen = DataGenerator(X_test, y_test, config["feature"]["sr"], config["feature"]["dt"],
                        n_classes=config["feature"]["n_classes"], batch_size=config["fit"]["batch_size"], n_channels=config["feature"]["n_channels"])

    # model
    model = models[args.model]
    
    model.summary()
    with open(f'{cur_dir}/{config["results_dir"]}/{model.name}_report.txt','w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

    plot_model(model, f'{cur_dir}/{config["results_dir"]}/{model.name}.png', show_shapes=True)
    plot_model(model, f'{cur_dir}/{config["results_dir"]}/{model.name}.pdf', show_shapes=True)

    # Callbacks
    # initialize tqdm callback with default parameters
    tqdm_cb = tfa.callbacks.TQDMProgressBar(leave_epoch_progress=True, leave_overall_progress=True)
    checkpoint_cb = ModelCheckpoint(f'{cur_dir}/{config["results_dir"]}/{model.name}.h5', save_best_only=True)
    early_stopping_cb = EarlyStopping(patience=5, restore_best_weights=True)
    model_name = f'{model.name}-{time.strftime("run_%Y_%m_%d-%H_%M_%S")}'
    tensorboard_cb = TensorBoard(log_dir = f'{cur_dir}/{config["log_dir"]}/{model_name}')

    start_time = datetime.now()
    history = model.fit(train_gen, 
                        steps_per_epoch=int(train_size / config["fit"]["batch_size"]), 
                        validation_data=valid_gen, 
                        epochs=config["fit"]["epochs"],
                        validation_steps=int(valid_size / config["fit"]["batch_size"]), 
                        verbose=0, 
                        callbacks=[tqdm_cb, checkpoint_cb, early_stopping_cb, tensorboard_cb])

    time_elapsed = datetime.now() - start_time 
    print(f'{model.name} train elapsed (hh:mm:ss.ms) {time_elapsed}')
    # plots the accuracy and loss for against epochs
    plot_history(history=history, dir=f'{cur_dir}/{config["results_dir"]}', file_name=f'{model.name}_history')

    device.reset()    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Malfunctioning Industrial Machine Investigation and Inspection (MIMII). Classification/Anomaly Detection Training.')
    parser.add_argument('--model', type=str, default='myconv2d',
                        help='model to train. (myconv1d, myconv2d, mylstm, myconv1dae, myconv2dae')
    args, _ = parser.parse_known_args()

    train(args)

