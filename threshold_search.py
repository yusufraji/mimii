from pathlib import Path

import tensorflow as tf
import yaml
from numba import cuda
from silence_tensorflow import silence_tensorflow
from tensorflow.keras.models import load_model

from utils import loss_dist

silence_tensorflow()

tf.config.experimental.list_physical_devices("GPU")

cur_dir = Path.cwd()
device = cuda.get_current_device()
# load config yaml
with open("config.yaml") as stream:
    config = yaml.safe_load(stream)


# plot the loss distribution of train, valid and test
model = load_model(cur_dir / "results" / "2d_convolution_autoencoder.h5")
loss_dist(
    model=model,
    results_dir=cur_dir / config["results_dir"],
    dataset_dir=cur_dir / "dataset",
)
device.reset()
