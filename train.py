import time
from pathlib import Path

from scipy.io import wavfile


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, TensorBoard

from models import MyConv2D
from utils import save_fig, model_metrics, plot_history


import yaml


# data generator
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, wav_paths, labels, sr, dt, n_classes, batch_size=32, shuffle=True):
        self.wav_paths = wav_paths
        self.labels = labels
        self.sr = sr
        self.dt = dt
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.shuffle = True
        self.on_epoch_end()


    def __len__(self):
        return int(np.floor(len(self.wav_paths) / self.batch_size))


    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        wav_paths = [self.wav_paths[k] for k in indexes]

        labels = [self.labels[k] for k in indexes]
        
        # generate a batch of time data
        X = np.empty((self.batch_size, int(self.sr*self.dt), 1), dtype=np.float32)
        Y = np.empty((self.batch_size, self.n_classes), dtype=np.float32)

        for i, (path, label) in enumerate(zip(wav_paths, labels)):
            rate, wav = wavfile.read(path)

            X[i,] = wav.reshape(-1, 1)
            Y[i,] = to_categorical(label, num_classes=self.n_classes)

        return X, Y


    def on_epoch_end(self):
        self.indexes = np.arange(len(self.wav_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)


# create train test and valid set
def train_test_valid(data):

    # integer encoder
    le = LabelEncoder()
    integer_encoded = le.fit_transform(data['machine_type'])
    # Split train, test, and valid set
    X_train_full, X_test, y_train_full, y_test = train_test_split(data['normal'], integer_encoded, test_size=0.2, shuffle=True, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=0.1, shuffle=True, random_state=42)

    return np.asarray(X_train), np.asarray(X_valid), np.asarray(X_test), y_train, y_valid, y_test


if __name__ == "__main__":
    
    # load config yaml
    with open("config.yaml") as stream:
        config = yaml.safe_load(stream)

    # create directories
    Path.mkdir(config["results_dir"], exist_ok=True)
    Path.mkdir(config["log_dir"], exist_ok=True)

    # setup result
    result_file = "{result}/{file}".format(result=config["results_dir"], file=config["result_file"])
    results = {}

    # fetch dataset
    data = pd.DataFrame(columns=['machine_type', 'machine_id', 'normal', 'abnormal', 'db'])
    for db in ["mindB"]:    # ["0dB", "6dB", "min6dB"]
        for folder in (config["dataset_dir"] / db).iterdir():
            for id in folder.iterdir():
                normal_glob = (id / 'normal' ).glob("*.wav")
                abnormal_glob = (id / "abnormal").glob("*.wav")
                for normal, abnormal in zip(normal_glob, abnormal_glob):
                    values = [folder.name, id.name, normal, abnormal, db]
                    a_dict = dict(zip(list(data.columns), values))
                    data = data.append(a_dict, ignore_index=True)


    # train test valid split
    X_train, X_valid, X_test, y_train, y_valid, y_test = train_test_valid(data)
    train_size = len(X_train)
    valid_size = len(X_valid)

    # dataset generator
    train_gen = DataGenerator(X_train, y_train, config["sr"], config["dt"],
                        n_classes=config["n_classes"], batch_size=config["batch_size"])

    valid_gen = DataGenerator(X_valid, y_valid, config["sr"], config["dt"],
                        n_classes=config["n_classes"], batch_size=config["batch_size"])

    test_gen = DataGenerator(X_test, y_test, config["sr"], config["dt"],
                        n_classes=config["n_classes"], batch_size=config["batch_size"])

    # model
    model = MyConv2D(N_CLASSES=config["n_classes"], SR=config["sr"], DT=config["dt"])

    model.summary()

    plot_model(model, f"{config["results_dir"]}{model.name}.png", show_shapes=True)

    # Callbacks
    # initialize tqdm callback with default parameters
    tqdm_cb = tfa.callbacks.TQDMProgressBar()
    checkpoint_cb = ModelCheckpoint(os.path.join(results_dir, f"{model.name}.h5"), save_best_only=True)
    early_stopping_cb = EarlyStopping(patience=5, restore_best_weights=True)
    model_name = f'{model.name}-{time.strftime("run_%Y_%m_%d-%H_%M_%S")}'
    tensorboard_cb = TensorBoard(log_dir = log_dir / model_name)

    history = model.fit(train_gen, 
                        steps_per_epoch=int(train_size / batch_size), 
                        validation_data=valid_gen, 
                        epochs=50,
                        validation_steps=int(valid_size / batch_size), 
                        verbose=1, 
                        callbacks=[checkpoint_cb, early_stopping_cb, tensorboard_cb])

    # plots the accuracy and loss for against epochs
    plot_history(history)