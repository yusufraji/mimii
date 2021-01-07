from pathlib import Path
from datetime import datetime
import seaborn as sns

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import scikitplot as skplt
import yaml
from sklearn.metrics import classification_report
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


def model_metrics(y_test, y_pred, decoded, model_name):
    print(f"Decoded classes after applying inverse of label encoder: {decoded}")

    skplt.metrics.plot_confusion_matrix(
        y_test,
        y_pred,
        labels=decoded,
        title_fontsize="large",
        text_fontsize="medium",
        cmap="Greens",
        figsize=(8, 6),
    )
    plt.show()

    print(
        "The classification report for the model : \n\n"
        + classification_report(y_test, y_pred)
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


def make_ae_predictions(model, X, y, show=True):
    if show:
        print("\n", X, "\t", y)
    # load the preprocessed audio (.npy) file
    X = np.load(X)
    y = np.load(y)
    # extend the input dimension of the model
    X = X[np.newaxis, :, :, :]
    y = y[np.newaxis, :, :, :]
    # predict and compute the loss
    y_pred = model.predict(X)
    #loss = np.mean((y - y_pred) ** 2)
    # loss = rmse(y, y_pred)
    loss = np.mean(np.abs(y - y_pred))
    if show:
        print(f"Loss: {loss}")

    return loss


def make_predictions(model, le, X, y, show=True):
    rate, wav = wavfile.read(X)
    wav = wav.reshape(1, wav.shape[0], wav.shape[1])
    y_pred = model.predict(wav)
    y_mean = np.mean(y_pred, axis=0)
    y_pred = np.argmax(y_mean)
    y_pred = le.inverse_transform([y_pred])
    if show:
        print(f"True class: {y}. Predicted class: {y_pred}")

    return y_pred

def loss_dist(model, results_dir, dataset_dir):
    """
    docstring
    """
    print("plotting loss distribution")
    train = pd.read_csv(dataset_dir / "train_ae_df.csv")
    valid = pd.read_csv(dataset_dir / "valid_ae_df.csv")
    test = pd.read_csv(dataset_dir / "test_ae_df.csv")
    # remove the other normal files set aside for testing
    # test = test.iloc[1982:]

    start_time = datetime.now()
    print('======= computing reconstruction loss for train data ======= ')
    train["loss"] = train.progress_apply(
        lambda x: make_ae_predictions(model, x.X_train, x.y_train, show=False), axis=1
    )
    print('======= computing reconstruction loss for valid data ======= ')
    valid["loss"] = valid.progress_apply(
        lambda x: make_ae_predictions(model, x.X_valid, x.y_valid, show=False), axis=1
    )
    print('======= computing reconstruction loss for test data ======= ')
    test["loss"] = test.progress_apply(
        lambda x: make_ae_predictions(model, x.X_test, x.y_test, show=False), axis=1
    )

    # train_pred = train_pred.reshape(train_pred.shape[0], train_pred.shape[2])
    # valid_pred = history.predict(valid)
    # valid_pred = valid_pred.reshape(valid_pred.shape[0], valid_pred.shape[2])
    # test_pred = history.predict(test)
    # test_pred = test_pred.reshape(test_pred.shape[0], test_pred.shape[2])

    threshold = train["loss"].quantile(0.95)

    plt.figure(figsize=(16, 9))
    sns.distplot(
        train["loss"], kde=True, label="train", color=sns.color_palette("bright")[0]
    )
    plt.vlines(threshold, 0, 5, linestyles ="dashed", colors ="k")
    plt.text(threshold, 5, f"95th Percentile @ {threshold}", size = 10, alpha =.8)
    sns.distplot(
        valid["loss"], kde=True, label="valid", color=sns.color_palette("bright")[1]
    )
    sns.distplot(
        test["loss"], kde=True, label="test", color=sns.color_palette("bright")[2]
    )
    plt.legend()
    plt.grid(True)
    plt.title(f"Loss Distribution of {model.name}.")
    save_fig(fig_dir=results_dir, fig_id=f"loss_dist_{model.name}")
    time_elapsed = datetime.now() - start_time
    print(
        f"loss distribution for train, valid, test, took {time_elapsed}(hh:mm:ss.ms)."
    )

def rmse(y_true, y_pred):
    """
    docstring
    """
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))
