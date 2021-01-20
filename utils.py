from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.io import wavfile

mpl.rc("axes", labelsize=14)
mpl.rc("xtick", labelsize=12)
mpl.rc("ytick", labelsize=12)
import numpy as np
import pandas as pd
import scikitplot as skplt
import seaborn as sns
import tensorflow as tf
import yaml
from sklearn.metrics import auc, classification_report, roc_curve
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

tqdm.pandas()
cur_dir = Path.cwd()

# load config yaml
with open("config.yaml") as stream:
    config = yaml.safe_load(stream)


def save_fig(fig_dir, fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = f"{fig_dir}/{fig_id}.{fig_extension}"
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def model_metrics(y_true, y_pred, decoded, model_name, fig_dir):
    print(f"Decoded classes after applying inverse of label encoder: {decoded}")

    skplt.metrics.plot_confusion_matrix(
        y_true,
        y_pred,
        labels=decoded,
        x_tick_rotation=90,
        normalize="true",
        title_fontsize="large",
        text_fontsize="medium",
        cmap="Greens",
        figsize=(8, 6),
    )
    plt.show()
    save_fig(fig_dir=fig_dir, fig_id=f"confusion_matrix_{model_name}")
    print(
        "The classification report for the model : \n\n"
        + classification_report(y_true, y_pred)
    )


# plots the accuracy and loss against epochs
def plot_history(history, dir, file_name, ae):

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epoch = history.epoch

    plt.figure(figsize=(11, 4))
    if not ae:
        acc = history.history["accuracy"]
        val_acc = history.history["val_accuracy"]

        plt.suptitle(f"History of {file_name}")
        plt.subplot(121)
        plt.plot(epoch, acc, label="Training accuracy", color="r")
        plt.plot(epoch, val_acc, label="Validation accuracy", color="b")
        plt.gca().set_ylim(0, 1)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.title("Training and Validation Accuracy")

        plt.subplot(122)
    plt.plot(epoch, loss, label="Training loss", color="r")
    plt.plot(epoch, val_loss, label="Validation loss", color="b")
    if not ae:
        plt.gca().set_ylim(0, 1)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.title("Training and Validation Loss")

    save_fig(dir, file_name)
    plt.show()


# fetch dataset
def fetch_dataset(extension="wav", dataset_file_name="dataset_df"):
    print(
        "================================= fetching dataset =================================\n"
    )
    data = pd.DataFrame(
        columns=[
            "machine_type",
            "machine_id",
            "machine_type_id",
            "normal",
            "abnormal",
            "db",
        ]
    )
    for db in tqdm(["0dB", "6dB", "min6dB"], desc="fetching db files.", leave=True):
        for folder in tqdm(
            (Path.cwd() / config["dataset_dir"] / db).iterdir(),
            desc=f"fetching machine files.",
            leave=False,
        ):
            for id in tqdm(
                folder.iterdir(), desc=f"fetching machine id files.", leave=False
            ):
                normal_glob = (id / "normal").glob(f"*.{extension}")
                abnormal_glob = (id / "abnormal").glob(f"*.{extension}")
                for normal, abnormal in zip(normal_glob, abnormal_glob):
                    values = [
                        folder.name,
                        id.name,
                        f"{folder.name}_{id.name}",
                        normal,
                        abnormal,
                        db,
                    ]
                    a_dict = dict(zip(list(data.columns), values))
                    data = data.append(a_dict, ignore_index=True)
    data.to_csv(cur_dir / config["dataset_dir"] / f"{dataset_file_name}.csv")
    return data


def make_ae_predictions(model, X, y, show=True, **kwargs):
    pred_class = None
    if show:
        print("\n", X, "\t", y)
    # assign the class to the file
    if Path(X).parts[-2] == "normal":
        true_class = "normal"
    elif Path(X).parts[-2] == "abnormal":
        true_class = "abnormal"
    # load the preprocessed audio (.npy) file
    X = np.load(X)
    y = np.load(y)
    # extend the input dimension of the model
    X = X[np.newaxis, :, :, :]
    y = y[np.newaxis, :, :, :]
    # predict and compute the loss
    y_pred = model.predict(X)
    # loss = np.mean((y - y_pred) ** 2)
    # loss = rmse(y, y_pred)
    loss = np.mean(np.abs(y - y_pred))
    if "threshold" in kwargs.keys():
        print(f'Threshold: {kwargs["threshold"]}')
        if loss <= kwargs["threshold"]:
            pred_class = "normal"
        elif loss > kwargs["threshold"]:
            pred_class = "abnormal"
    if show:
        print(f"Loss: {loss}, True class: {true_class}, Predicted class: {pred_class}")

    return pd.Series([loss, true_class, pred_class])


def make_predictions(model, le, X, y, show=True):
    rate, wav = wavfile.read(X)
    wav = wav.reshape(1, wav.shape[0], wav.shape[1])
    y_pred = model.predict(wav)
    y_mean = np.mean(y_pred, axis=0)
    y_pred = np.argmax(y_mean)
    y_pred = le.inverse_transform([y_pred])
    y_pred = str(np.squeeze(y_pred))
    if show:
        print(f"True class: {y}. Predicted class: {y_pred}")

    return y_pred


def loss_dist(model, results_dir, dataset_dir, id):
    """
    docstring
    """
    print("plotting loss distribution")
    train = pd.read_csv(dataset_dir / f"train_{id}.csv")
    valid = pd.read_csv(dataset_dir / f"valid_{id}.csv")
    test = pd.read_csv(dataset_dir / f"test_{id}.csv")

    # move normal files set aside for testing from test dataframe to train dataframe
    new_df = test.loc[
        test["X_test"].apply(lambda x: str(Path(x).parts[-2]) == "normal")
    ]
    new_df.columns = train.columns
    train = pd.concat([train, new_df], ignore_index=True)
    test = test.loc[
        test["X_test"].apply(lambda x: str(Path(x).parts[-2]) == "abnormal")
    ]

    start_time = datetime.now()
    print("======= computing reconstruction loss for train data ======= ")
    train[["loss", "true_class", "pred_class"]] = train.progress_apply(
        lambda x: make_ae_predictions(model, x.X_train, x.y_train, show=False), axis=1
    )
    print("======= computing reconstruction loss for valid data ======= ")
    valid[["loss", "true_class", "pred_class"]] = valid.progress_apply(
        lambda x: make_ae_predictions(model, x.X_valid, x.y_valid, show=False), axis=1
    )
    print("======= computing reconstruction loss for test data ======= ")
    test[["loss", "true_class", "pred_class"]] = test.progress_apply(
        lambda x: make_ae_predictions(model, x.X_test, x.y_test, show=False), axis=1
    )

    # compute/plot roc, auc...
    data = pd.concat(
        [
            train.rename(columns={"X_train": "X", "y_train": "y"}),
            valid.rename(columns={"X_valid": "X", "y_valid": "y"}),
            test.rename(columns={"X_test": "X", "y_test": "y"}),
        ],
        ignore_index=True,
    )
    # binarize the true classes
    lb_true = label_binarize(data["true_class"], classes=["normal", "abnormal"])
    data_fpr, data_tpr, data_thresholds = roc_curve(
        lb_true,
        data["loss"],
    )
    data_roc_auc = auc(
        data_fpr,
        data_tpr,
    )

    plt.figure(figsize=(9, 9))
    plt.plot(data_fpr, data_tpr, lw=2, label=f"AUC = {data_roc_auc:.4f}", alpha=0.8)
    plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
    plt.xlim([-0.001, 1])
    plt.ylim([0, 1.001])
    plt.ylabel("True Positive Rate (Positive label: 1)")
    plt.xlabel("False Positive Rate (Positive label: 1)")
    plt.legend(loc="lower right")
    plt.title(f"ROC curve of {model.name}.")
    save_fig(fig_dir=results_dir, fig_id=f"roc_{model.name}")

    # find threshold and plot reconstruction loss distribution
    train_threshold = train["loss"].quantile(0.95)
    train_threshold_std = np.mean(train["loss"]) + np.std(train["loss"])
    test_threshold = test["loss"].quantile(0.95)
    test_threshold_std = np.mean(test["loss"]) + np.std(test["loss"])

    plt.figure(figsize=(16, 9))
    sns.distplot(
        train["loss"], kde=True, label="train", color=sns.color_palette("bright")[0]
    )
    plt.vlines(train_threshold, 0, 5.5, linestyles="dashed", colors="k")
    plt.text(
        train_threshold,
        5.5,
        f"train threshold, 95th Percentile @ {train_threshold:.3f}",
        size=10,
        alpha=0.8,
    )
    plt.vlines(train_threshold_std, 0, 6, linestyles="dashed", colors="r")
    plt.text(
        train_threshold_std,
        6,
        f"train threshold, one std above mean @ {train_threshold_std:.3f}",
        size=10,
        alpha=0.8,
    )
    sns.distplot(
        valid["loss"], kde=True, label="valid", color=sns.color_palette("bright")[1]
    )
    sns.distplot(
        test["loss"], kde=True, label="test", color=sns.color_palette("bright")[2]
    )
    plt.vlines(test_threshold, 0, 4.5, linestyles="dashdot", colors="k")
    plt.text(
        test_threshold,
        4.5,
        f"test threshold, 95th Percentile @ {test_threshold:.3f}",
        size=10,
        alpha=0.8,
    )
    plt.vlines(test_threshold_std, 0, 5, linestyles="dashdot", colors="r")
    plt.text(
        test_threshold_std,
        5,
        f"test threshold, one std above mean @ {test_threshold_std:.3f}",
        size=10,
        alpha=0.8,
    )
    plt.legend()
    plt.grid(True)
    plt.title(f"Loss Distribution of {model.name}.")
    save_fig(fig_dir=results_dir, fig_id=f"loss_dist_{model.name}")
    time_elapsed = datetime.now() - start_time
    print(
        f"{id} loss distribution for train, valid, test, took {time_elapsed}(hh:mm:ss.ms)."
    )

    return train_threshold_std


def rmse(y_true, y_pred):
    """
    docstring
    """
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))


# define your custom callback for loss distribution prediction of train and test
# data
class LossDistributionCallback(tf.keras.callbacks.Callback):
    # reference: https://stackoverflow.com/a/61686275
    def __init__(self, dataset_dir, id):
        super(LossDistributionCallback, self).__init__()
        self.dataset_dir = dataset_dir
        self.id = id

    def on_epoch_end(self, epoch, logs={}):
        logs["threshold_diff"] = float("-inf")
        train = pd.read_csv(self.dataset_dir / f"train_{self.id}.csv")
        test = pd.read_csv(self.dataset_dir / f"test_{self.id}.csv")

        # predict the train and test data
        train["loss"] = train.progress_apply(
            lambda x: self.evaluate_model(x.X_train, x.y_train), axis=1
        )
        test["loss"] = test.progress_apply(
            lambda x: self.evaluate_model(x.X_test, x.y_test), axis=1
        )

        # compute the thresholds
        train_threshold_std = np.mean(train["loss"]) + np.std(train["loss"])
        test_threshold_std = np.mean(test["loss"]) + np.std(test["loss"])
        # get the diff between test and train threshold.
        # the training objective is to make the diff as high as possible
        # test threshold must be higher that train threshold
        threshold_diff = test_threshold_std - train_threshold_std
        print(f"threshold diff: {threshold_diff} at epoch: {epoch}")
        logs["threshold_diff"] = threshold_diff

    def evaluate_model(self, X, y):
        # load the preprocessed audio (.npy) file
        X = np.load(X)
        y = np.load(y)
        # extend the input dimension of the model
        X = X[np.newaxis, :, :, :]
        y = y[np.newaxis, :, :, :]
        # predict and compute the loss
        y_pred = self.model.predict(X)
        loss = np.mean(np.abs(y - y_pred))
        return loss
