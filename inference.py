import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile
from silence_tensorflow import silence_tensorflow

silence_tensorflow()

import tensorflow as tf
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import LabelEncoder

tf.config.experimental.list_physical_devices("GPU")
import yaml
from kapre.time_frequency import STFT, ApplyFilterbank, Magnitude, MagnitudeToDecibel
from numba import cuda
from tensorflow.keras.metrics import MeanSquaredError as mse
from tensorflow.keras.models import load_model

from utils import make_ae_predictions, make_predictions, model_metrics, rmse


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
        "myconv2d": load_model(
            results_dir / "2d_convolution.h5",
            custom_objects={
                "STFT": STFT,
                "Magnitude": Magnitude,
                "ApplyFilterbank": ApplyFilterbank,
                "MagnitudeToDecibel": MagnitudeToDecibel,
            },
        ),
        "myconv2dae": None,
    }
    assert args.model in models.keys(), f"{args.model} is unavailable."
    assert config["feature"]["n_classes"] in [
        4,
        16,
    ], f'n_classes({config["feature"]["n_classes"]}) must either be 4 or 16'

    if args.model == "myconv2dae":
        # redirect console output to txt file
        sys.stdout = open(Path(config["logs_dir"]) / "infer_autoencoder.txt", "w")
        # load the test set
        assert "test_ae_df.csv" in [
            x.name for x in (cur_dir / config["dataset_dir"]).iterdir()
        ], f"test_df.csv not found!"

        result_file = cur_dir / config["results_dir"] / config["result_file"]
        with open(result_file, "r") as file:
            train_results = yaml.safe_load(file)

        dataset_df = pd.read_csv(cur_dir / config["dataset_dir"] / "dataset_df.csv")

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
                    custom_objects={"rmse": rmse},
                )

                start_time = datetime.now()
                test_data[["loss", "true_class", "pred_class"]] = test_data.apply(
                    lambda x: make_ae_predictions(
                        model,
                        x.X_test,
                        x.y_test,
                        threshold=train_results[id]["threshold"],
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
                le = LabelEncoder()
                test_fpr, test_tpr, test_thresholds = roc_curve(
                    le.fit_transform(test_data["true_class"]),
                    test_data["loss"],
                )
                train_roc_auc = auc(test_fpr, test_tpr)

                time_elapsed = datetime.now() - start_time
                print(
                    f"Predicted {test_data.shape[0]} with {model.name} in {time_elapsed}(hh:mm:ss.ms)."
                )

                test_data.to_csv(
                    dataset_dir / f"{id}_prediction.csv",
                    index=False,
                )

        sys.stdout.close()

    elif args.model == "myconv2d":
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
        default="myconv2dae",
        help="model to train. (myconv1d, myconv2d, mylstm, myconv1dae, myconv2dae",
    )
    args, _ = parser.parse_known_args()

    infer(args)
