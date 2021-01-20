import argparse
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_addons as tfa
import yaml
from librosa.core.spectrum import amplitude_to_db
from librosa.feature.spectral import melspectrogram
from numba import cuda
from scipy.io import wavfile
from silence_tensorflow import silence_tensorflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import (
    EarlyStopping,
    LambdaCallback,
    LearningRateScheduler,
    ModelCheckpoint,
    TensorBoard,
)
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.python.keras.engine.training import Model
from tqdm import tqdm

from models import MyConv2D, MyConv2DAE
from utils import (
    LossDistributionCallback,
    fetch_dataset,
    loss_dist,
    make_ae_predictions,
    make_predictions,
    model_metrics,
    plot_history,
    save_fig,
)

# silence_tensorflow()

tf.config.experimental.list_physical_devices("GPU")
physical_devices = tf.config.list_physical_devices("GPU")
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass
warnings.filterwarnings(action="ignore")


class DataGeneratorAE(tf.keras.utils.Sequence):
    def __init__(
        self,
        npy_paths,
        sr,
        dt,
        batch_size=32,
        n_channels=1,
        n_mels=128,
        hop_length=512,
        shuffle=True,
    ):
        # 'Initialization'
        self.npy_paths = npy_paths
        self.sr = sr
        self.dt = dt
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        # 'Denotes the number of batches per epoch'
        return int(np.floor(len(self.npy_paths) / self.batch_size))

    def __getitem__(self, index):
        # 'Generate one batch of data'
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        npy_paths = [self.npy_paths[k] for k in indexes]

        # generate a batch of time data. X = Y = [32,128,313,8]
        # reference:
        # https://stackoverflow.com/questions/51241499/parameters-to-control-the-size-of-a-spectrogram
        n_frames = int(np.floor((self.sr * self.dt) / self.hop_length) + 1)
        # n_frames must be even for autoencoder symmetry
        # round n_frames up to the nearest even value!
        n_frames = int(np.ceil(n_frames / 2.0) * 2)

        X = np.empty(
            (self.batch_size, self.n_mels, n_frames, self.n_channels), dtype=np.float32
        )
        Y = np.empty(
            (self.batch_size, self.n_mels, n_frames, self.n_channels), dtype=np.float32
        )
        for i, path in enumerate(npy_paths):
            S_db = np.load(path)

            X[
                i,
            ] = S_db
            Y[
                i,
            ] = S_db

        return X, Y

    def on_epoch_end(self):
        # 'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.npy_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)


# data generator
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        wav_paths,
        labels,
        sr,
        dt,
        n_classes,
        batch_size=32,
        n_channels=1,
        shuffle=True,
    ):
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
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        wav_paths = [self.wav_paths[k] for k in indexes]

        labels = [self.labels[k] for k in indexes]

        # generate a batch of time data
        X = np.empty(
            (self.batch_size, int(self.sr * self.dt), self.n_channels), dtype=np.float32
        )
        Y = np.empty((self.batch_size, self.n_classes), dtype=np.float32)

        for i, (path, label) in enumerate(zip(wav_paths, labels)):
            rate, wav = wavfile.read(path)

            X[
                i,
            ] = wav[:, : self.n_channels]
            Y[
                i,
            ] = to_categorical(label, num_classes=self.n_classes)

        return X, Y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.wav_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)


# create train test and valid set
def train_test_valid(data, n_classes=16, test_size=0.2, valid_size=0.1, ae=False):

    if ae:
        # Split train, test, and valid set
        X_train_full, X_test = train_test_split(
            data["normal"], test_size=test_size, shuffle=True, random_state=42
        )
        X_train, X_valid = train_test_split(
            X_train_full, test_size=valid_size, shuffle=True, random_state=42
        )

        # append the abnormal files to X_test
        X_test = X_test.values.tolist()
        X_test.extend(data["abnormal"].values.tolist())

        y_train, y_valid, y_test = X_train, X_valid, X_test

        train_df = pd.DataFrame()
        train_df["X_train"], train_df["y_train"] = X_train, y_train

        valid_df = pd.DataFrame()
        valid_df["X_valid"], valid_df["y_valid"] = X_valid, y_valid

        test_df = pd.DataFrame()
        test_df["X_test"], test_df["y_test"] = X_test, y_test

    else:
        # integer encoder
        le = LabelEncoder()
        if n_classes == 4:
            integer_encoded = le.fit_transform(data["machine_type"])
        else:
            integer_encoded = le.fit_transform(data["machine_type_id"])
        # Split train, test, and valid set
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            data["normal"],
            integer_encoded,
            test_size=test_size,
            shuffle=True,
            random_state=42,
        )
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train_full,
            y_train_full,
            test_size=valid_size,
            shuffle=True,
            random_state=42,
        )

        train_df = pd.DataFrame()
        train_df["X_train"], train_df["y_train"] = X_train, le.inverse_transform(
            y_train
        )

        valid_df = pd.DataFrame()
        valid_df["X_valid"], valid_df["y_valid"] = X_valid, le.inverse_transform(
            y_valid
        )

        test_df = pd.DataFrame()
        test_df["X_test"], test_df["y_test"] = X_test, le.inverse_transform(y_test)

    return (
        np.asarray(X_train),
        np.asarray(X_valid),
        np.asarray(X_test),
        y_train,
        y_valid,
        y_test,
        train_df,
        valid_df,
        test_df,
    )


def fit_model(
    model,
    config,
    id,
    ae,
    train_size,
    valid_size,
    train_gen,
    valid_gen,
    dataset_dir,
    results_dir,
    logs_dir,
):
    """
    docstring
    """
    model.summary()
    with open(f"{results_dir}/{model.name}_report.txt", "w") as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + "\n"))

    plot_model(model, f"{results_dir}/{model.name}.png", show_shapes=True)
    plot_model(model, f"{results_dir}/{model.name}.pdf", show_shapes=True)

    # Callbacks
    # initialize tqdm callback with default parameters
    tqdm_cb = tfa.callbacks.TQDMProgressBar(
        leave_epoch_progress=True, leave_overall_progress=True
    )
    checkpoint_cb = ModelCheckpoint(
        filepath=f"{results_dir}/{model.name}.h5",
        monitor="threshold_diff",
        verbose=1,
        save_best_only=True,
        mode="min",
    )
    early_stopping_cb = EarlyStopping(
        monitor="threshold_diff",
        patience=10,
        verbose=2,
        mode="max",
        restore_best_weights=True,
    )
    model_name = f'{model.name}-{time.strftime("run_%Y_%m_%d-%H_%M_%S")}'
    tensorboard_cb = TensorBoard(log_dir=f"{logs_dir}/{model_name}")
    loss_dist_cb = LossDistributionCallback(dataset_dir, id)
    # Stream the epoch loss to a file.
    txt_log = open(f"{logs_dir}_loss_log.txt", mode="wt", buffering=1)

    def model_best(logs, txt_log):
        """
        docstring
        """
        txt_log.write(
            f'model keys: {logs.keys()}, loss: {logs["loss"]}\n',
        )
        txt_log.close()

    save_op_cb = LambdaCallback(
        on_epoch_end=lambda epoch, logs: txt_log.write(
            f'model keys: {logs.keys()}, epoch: {epoch}, loss: {logs["loss"]}, threshold_diff: {logs["threshold_diff"]}\n'
        ),
        on_train_end=lambda logs: model_best(logs, txt_log),
    )

    start_time = datetime.now()
    history = model.fit(
        train_gen,
        steps_per_epoch=int(train_size / config["fit"]["batch_size"]),
        validation_data=valid_gen,
        epochs=config["fit"]["epochs"],
        validation_steps=int(valid_size / config["fit"]["batch_size"]),
        verbose=2,
        callbacks=[
            # tqdm_cb,
            loss_dist_cb,
            early_stopping_cb,
            # checkpoint_cb,
            tensorboard_cb,
            save_op_cb,
        ],
    )
    model.save(f"{results_dir}/{model.name}.h5")

    time_elapsed = datetime.now() - start_time
    print(f"{model.name} train elapsed (hh:mm:ss.ms) {time_elapsed}")
    # plots the accuracy and loss for against epochs
    plot_history(
        history=history,
        dir=f"{results_dir}",
        file_name=f"{model.name}_history",
        ae=ae,
    )
    return model


def train(args):
    """
    docstring
    """
    global cur_dir
    cur_dir = Path.cwd()
    device = cuda.get_current_device()
    # load config yaml
    with open("config.yaml") as stream:
        config = yaml.safe_load(stream)

    models = {
        "myconv2d": MyConv2D(
            N_CLASSES=config["feature"]["n_classes"],
            SR=config["feature"]["sr"],
            DT=config["feature"]["dt"],
            N_CHANNELS=config["feature"]["n_channels"],
        ),
        "myconv2dae": MyConv2DAE(
            ID="2d_convolution_autoencoder",
            N_CHANNELS=config["feature"]["n_channels"],
            SR=config["feature"]["sr"],
            DT=config["feature"]["dt"],
            N_MELS=config["feature"]["n_mels"],
            HOP_LENGTH=config["feature"]["hop_length"],
        ),
    }
    assert args.model in models.keys(), f"{args.model} is unavailable."
    assert config["feature"]["n_classes"] in [
        4,
        16,
    ], f'n_classes({config["feature"]["n_classes"]}) must either be 4 or 16'

    # create directories
    Path.mkdir(cur_dir / config["results_dir"], parents=True, exist_ok=True)
    Path.mkdir(cur_dir / config["logs_dir"], parents=True, exist_ok=True)

    # set y_train, y_valid and y_test to X_train, X_valid and X_test if model is
    # an autoencoder, and create a new data generatore for ae
    if args.model == "myconv2dae":
        # redirect console output to txt file
        sys.stdout = open(Path(config["logs_dir"]) / "train_autoencoder.txt", "w")

        # setup result
        result_file = "{result}/{file}".format(
            result=config["results_dir"], file=config["result_file"]
        )
        results = {}

        start_time = datetime.now()
        ae = True
        # fetch dataset
        if "dataset_ae_df.csv" in [
            x.name for x in (cur_dir / config["dataset_dir"]).iterdir()
        ]:
            full_data = pd.read_csv(
                cur_dir / config["dataset_dir"] / "dataset_ae_df.csv"
            )
        else:
            full_data = fetch_dataset(
                extension="npy", dataset_file_name="dataset_ae_df"
            )

        for db in np.unique(full_data.db):
            for machine_type_id in np.unique(full_data.machine_type_id):

                id = f"{machine_type_id}_{db}"

                print("\n\n")
                print("=" * 20, end="")
                print(f" training {id} ", end="")
                print("=" * 20)

                dataset_dir = cur_dir / config["dataset_dir"] / db / machine_type_id
                results_dir = cur_dir / config["results_dir"] / db / machine_type_id
                logs_dir = cur_dir / config["logs_dir"] / db / machine_type_id
                # create directories
                Path.mkdir(dataset_dir, parents=True, exist_ok=True)
                Path.mkdir(results_dir, parents=True, exist_ok=True)
                Path.mkdir(logs_dir, parents=True, exist_ok=True)

                data = full_data.loc[
                    (full_data["machine_type_id"] == machine_type_id)
                    & (full_data["db"] == db)
                ]
                # train test valid split
                (
                    X_train,
                    X_valid,
                    X_test,
                    y_train,
                    y_valid,
                    y_test,
                    train_df,
                    valid_df,
                    test_df,
                ) = train_test_valid(
                    data,
                    n_classes=config["feature"]["n_classes"],
                    test_size=config["fit"]["test_size"],
                    valid_size=config["fit"]["valid_size"],
                    ae=True,
                )
                train_size = len(X_train)
                valid_size = len(X_valid)

                train_df.to_csv(dataset_dir / f"train_{id}.csv", index=False)

                valid_df.to_csv(dataset_dir / f"valid_{id}.csv", index=False)

                test_df.to_csv(dataset_dir / f"test_{id}.csv", index=False)

                train_gen = DataGeneratorAE(
                    X_train,
                    config["feature"]["sr"],
                    config["feature"]["dt"],
                    batch_size=config["fit"]["batch_size"],
                    n_channels=config["feature"]["n_channels"],
                    n_mels=config["feature"]["n_mels"],
                    hop_length=config["feature"]["hop_length"],
                )

                valid_gen = DataGeneratorAE(
                    X_valid,
                    config["feature"]["sr"],
                    config["feature"]["dt"],
                    batch_size=config["fit"]["batch_size"],
                    n_channels=config["feature"]["n_channels"],
                    n_mels=config["feature"]["n_mels"],
                    hop_length=config["feature"]["hop_length"],
                )

                test_gen = DataGeneratorAE(
                    X_test,
                    config["feature"]["sr"],
                    config["feature"]["dt"],
                    batch_size=config["fit"]["batch_size"],
                    n_channels=config["feature"]["n_channels"],
                    n_mels=config["feature"]["n_mels"],
                    hop_length=config["feature"]["hop_length"],
                )

                # fit the model
                model = MyConv2DAE(
                    ID=id,
                    N_CHANNELS=config["feature"]["n_channels"],
                    SR=config["feature"]["sr"],
                    DT=config["feature"]["dt"],
                    N_MELS=config["feature"]["n_mels"],
                    HOP_LENGTH=config["feature"]["hop_length"],
                )
                fitted_model = fit_model(
                    model=model,
                    config=config,
                    id=id,
                    ae=ae,
                    train_size=train_size,
                    valid_size=valid_size,
                    train_gen=train_gen,
                    valid_gen=valid_gen,
                    dataset_dir=dataset_dir,
                    results_dir=results_dir,
                    logs_dir=logs_dir,
                )
                # plot the loss distribution of train, valid and test
                threshold = loss_dist(
                    model=fitted_model,
                    results_dir=results_dir,
                    dataset_dir=dataset_dir,
                    id=id,
                )
                results[id] = {"threshold": float(np.round(threshold, 4))}

        # write results to yaml file
        with open(result_file, "w") as file:
            yaml.dump(results, file, default_flow_style=False)
        time_elapsed = datetime.now() - start_time
        print(f"{args.model} elapsed (hh:mm:ss.ms) {time_elapsed}")
        sys.stdout.close()

    # dataset generator
    elif args.model == "myconv2d":
        # redirect console output to txt file
        sys.stdout = open(Path(config["logs_dir"]) / "train_classifier.txt", "w")

        ae = False
        results_dir = cur_dir / config["results_dir"] / "myconv2d"
        logs_dir = cur_dir / config["logs_dir"] / "myconv2d"
        # create directories
        Path.mkdir(results_dir, parents=True, exist_ok=True)
        Path.mkdir(logs_dir, parents=True, exist_ok=True)

        # fetch dataset
        if "dataset_df.csv" in [
            x.name for x in (cur_dir / config["dataset_dir"]).iterdir()
        ]:
            data = pd.read_csv(cur_dir / config["dataset_dir"] / "dataset_df.csv")
        else:
            data = fetch_dataset(extension="wav", dataset_file_name="dataset_df")

        # train test valid split
        (
            X_train,
            X_valid,
            X_test,
            y_train,
            y_valid,
            y_test,
            train_df,
            valid_df,
            test_df,
        ) = train_test_valid(
            data,
            n_classes=config["feature"]["n_classes"],
            test_size=config["fit"]["test_size"],
            valid_size=config["fit"]["valid_size"],
        )
        train_size = len(X_train)
        valid_size = len(X_valid)

        train_df.to_csv(results_dir / "train_df.csv", index=False)

        valid_df.to_csv(results_dir / "valid_df.csv", index=False)

        test_df.to_csv(results_dir / "test_df.csv", index=False)

        train_gen = DataGenerator(
            X_train,
            y_train,
            config["feature"]["sr"],
            config["feature"]["dt"],
            n_classes=config["feature"]["n_classes"],
            batch_size=config["fit"]["batch_size"],
            n_channels=config["feature"]["n_channels"],
        )

        valid_gen = DataGenerator(
            X_valid,
            y_valid,
            config["feature"]["sr"],
            config["feature"]["dt"],
            n_classes=config["feature"]["n_classes"],
            batch_size=config["fit"]["batch_size"],
            n_channels=config["feature"]["n_channels"],
        )

        test_gen = DataGenerator(
            X_test,
            y_test,
            config["feature"]["sr"],
            config["feature"]["dt"],
            n_classes=config["feature"]["n_classes"],
            batch_size=config["fit"]["batch_size"],
            n_channels=config["feature"]["n_channels"],
        )

        # fit the model
        model = fit_model(
            model=models[args.model],
            config=config,
            id=args.model,
            ae=ae,
            train_size=train_size,
            valid_size=valid_size,
            train_gen=train_gen,
            valid_gen=valid_gen,
            results_dir=results_dir,
            logs_dir=logs_dir,
        )
        sys.stdout.close()

    device.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Malfunctioning Industrial Machine Investigation and Inspection (MIMII). Classification/Anomaly Detection Training."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="myconv2dae",
        help="model to train. (myconv1d, myconv2d, mylstm, myconv1dae, myconv2dae",
    )
    args, _ = parser.parse_known_args()

    train(args)
