import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc("axes", labelsize=14)
mpl.rc("xtick", labelsize=12)
mpl.rc("ytick", labelsize=12)

import numpy as np
import pandas as pd
from silence_tensorflow import silence_tensorflow
from sklearn.preprocessing import LabelEncoder

silence_tensorflow()

import tensorflow as tf
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize

tf.config.experimental.list_physical_devices("GPU")
import yaml
from kapre.time_frequency import STFT, ApplyFilterbank, Magnitude, MagnitudeToDecibel
from numba import cuda
from tensorflow.keras.models import load_model

from utils import make_ae_predictions, make_predictions, model_metrics, save_fig



def infer(args):
    """
    docstring
    """
    cur_dir = Path.cwd()
    results_dir = cur_dir / "results"
    device = cuda.get_current_device()
    # load config yaml
    with open("config.yaml") as stream:
        config = yaml.safe_load(stream)

    models = {
        "classifier": load_model(
            results_dir / "2d_convolution.h5",
            custom_objects={
                "STFT": STFT,
                "Magnitude": Magnitude,
                "ApplyFilterbank": ApplyFilterbank,
                "MagnitudeToDecibel": MagnitudeToDecibel,
            },
        ),
        "anomaly_detector": None,
    }
    assert args.model in models.keys(), f"{args.model} is unavailable."
    assert config["feature"]["n_classes"] in [
        4,
        16,
    ], f'n_classes({config["feature"]["n_classes"]}) must either be 4 or 16'

    if args.model == "anomaly_detector":
        # redirect console output to txt file
        sys.stdout = open(Path(config["logs_dir"]) / "infer_autoencoder.txt", "w")
        # load the test set
        assert "test_ae_df.csv" in [
            x.name for x in (cur_dir / config["dataset_dir"]).iterdir()
        ], f"test_df.csv not found!"

        result_file = cur_dir / config["results_dir"] / config["result_file"]
        with open(result_file, "r") as file:
            results = yaml.safe_load(file)

        dataset_df = pd.read_csv(cur_dir / config["dataset_dir"] / "dataset_df.csv")

        inference_start_time = datetime.now()
        for db in np.unique(dataset_df.db):
            for machine_type_id in np.unique(dataset_df.machine_type_id):

                id = f"{machine_type_id}_{db}"

                print("\n\n")
                print("=" * 20, end="")
                print(f" inferring {id} ", end="")
                print("=" * 20)

                dataset_dir = cur_dir / config["dataset_dir"] / db / machine_type_id
                results_dir = cur_dir / config["results_dir"] / db / machine_type_id

                train_data = pd.read_csv(dataset_dir / f"train_{id}.csv")
                test_data = pd.read_csv(dataset_dir / f"test_{id}.csv")
                model = load_model(
                    results_dir / f"{id}.h5",
                )

                start_time = datetime.now()
                test_data[["loss", "true_class", "pred_class"]] = test_data.apply(
                    lambda x: make_ae_predictions(
                        model,
                        x.X_test,
                        x.y_test,
                        threshold=results[id]["threshold"]["one_std"],
                    ),
                    axis=1,
                )
                model_metrics(
                    y_true=test_data["true_class"],
                    y_pred=test_data["pred_class"],
                    decoded=["normal", "abnormal"],
                    model_name=model.name,
                    fig_dir=results_dir,
                )
                # Binarize the true and predicted classes
                lb_true = label_binarize(
                    test_data["true_class"], classes=["normal", "abnormal"]
                )
                lb_pred = label_binarize(
                    test_data["pred_class"], classes=["normal", "abnormal"]
                )
                test_fpr, test_tpr, test_thresholds = roc_curve(
                    lb_true,
                    lb_pred,
                )
                roc_auc = auc(test_fpr, test_tpr)

                # plot the ROC curve
                plt.figure(figsize=(9, 9))
                plt.plot(
                    test_fpr, test_tpr, lw=2, label=f"AUC = {roc_auc:.4f}", alpha=0.8
                )
                plt.plot(
                    [0, 1],
                    [0, 1],
                    linestyle="--",
                    lw=2,
                    color="r",
                    label="Chance",
                    alpha=0.8,
                )
                plt.xlim([-0.001, 1])
                plt.ylim([0, 1.001])
                plt.ylabel("True Positive Rate (Positive label: 1)")
                plt.xlabel("False Positive Rate (Positive label: 1)")
                plt.legend(loc="lower right")
                plt.title(f"ROC curve of {model.name}.")
                save_fig(fig_dir=results_dir, fig_id=f"inference_roc_{model.name}")

                time_elapsed = datetime.now() - start_time
                print(
                    f"Predicted {test_data.shape[0]} with {model.name} in {time_elapsed}(hh:mm:ss.ms)."
                )

                test_data.to_csv(
                    dataset_dir / f"{id}_prediction.csv",
                    index=False,
                )
                results[id].update(
                    {
                        "roc_auc": float(np.round(roc_auc, 4)),
                    }
                )

        total_time_elapsed = datetime.now() - inference_start_time
        print(f"inference elapsed (hh:mm:ss.ms) {total_time_elapsed}")
        # write results to yaml file
        with open(result_file, "w") as file:
            yaml.dump(results, file, default_flow_style=False)
        sys.stdout.close()

    elif args.model == "classifier":
        # redirect console output to txt file
        sys.stdout = open(Path(config["logs_dir"]) / "infer_classifier.txt", "w")
        # load the test set
        assert "test_df.csv" in [
            x.name for x in (cur_dir / config["dataset_dir"]).iterdir()
        ], f"test_df.csv not found!"
        test_df = pd.read_csv(cur_dir / config["dataset_dir"] / "test_df.csv")

        le = LabelEncoder()
        encoded = le.fit_transform(test_df["y_test"])

        model = models[args.model]

        start_time = datetime.now()
        test_df["pred"] = test_df.apply(
            lambda x: make_predictions(model, le, x.X_test, x.y_test), axis=1
        )
        model_metrics(
            y_true=test_df["y_test"],
            y_pred=test_df["pred"],
            decoded=np.unique(le.inverse_transform(encoded)),
            model_name=model.name,
            fig_dir=results_dir,
        )
        time_elapsed = datetime.now() - start_time
        print(
            f"Predicted {test_df.shape[0]} with {model.name} in {time_elapsed}(hh:mm:ss.ms)."
        )

        test_df.to_csv(
            cur_dir / config["dataset_dir"] / f"{args.model}_prediction.csv",
            index=False,
        )
        sys.stdout.close()

    device.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Malfunctioning Industrial Machine Investigation and Inspection (MIMII). Classification/Anomaly Detection Inference."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="anomaly_detector",
        help="model to train. (classifier, anomaly_detector",
    )
    args, _ = parser.parse_known_args()

    infer(args)
