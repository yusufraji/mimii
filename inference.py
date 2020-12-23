import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.io import wavfile
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
tf.config.experimental.list_physical_devices('GPU')
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanSquaredError as mse
from kapre.time_frequency import STFT, Magnitude, ApplyFilterbank, MagnitudeToDecibel
from numba import cuda 

import yaml

def make_ae_predictions(model, X_test, y_test):
    print("\n", X_test, "\t", y_test)
    # load the preprocessed audio (.npy) file
    X_test = np.load(X_test)
    y_test = np.load(y_test)
    # extend the input dimension of the model
    X_test = X_test[np.newaxis,:,:,:]
    y_test = y_test[np.newaxis,:,:,:]
    # predict and compute the loss
    y_pred = model.predict(X_test)
    loss = np.mean((y_test-y_pred)**2)
    print(f'Loss: {loss}')

    return loss

def make_predictions(model, le, X_test, y_test):
    rate, wav = wavfile.read(X_test)
    wav = wav.reshape(1, wav.shape[0], wav.shape[1])
    y_pred = model.predict(wav)
    y_mean = np.mean(y_pred, axis=0)
    y_pred = np.argmax(y_mean)
    y_pred = le.inverse_transform([y_pred])
    print(f'True class: {y_test}. Predicted class: {y_pred}')

    return y_pred

def infer(args):
    """
    docstring
    """
    cur_dir = Path.cwd()
    results_dir = cur_dir / 'results'
    device = cuda.get_current_device()
    # load config yaml
    with open("config.yaml") as stream:
        config = yaml.safe_load(stream)

    models={'myconv2d' : load_model(results_dir / '2d_convolution.h5', custom_objects={'STFT':STFT, 'Magnitude':Magnitude, 'ApplyFilterbank':ApplyFilterbank, 'MagnitudeToDecibel':MagnitudeToDecibel}),
            'myconv2dae' : load_model(results_dir / '2d_convolution_autoencoder.h5')
            }
    assert args.model in models.keys(), f'{args.model} is unavailable.'
    assert config["feature"]["n_classes"] in [4, 16], f'n_classes({config["feature"]["n_classes"]}) must either be 4 or 16'

    if (args.model == 'myconv2dae'):
        # load the test set
        assert 'test_ae_df.csv' in [x.name for x in (cur_dir / config["dataset_dir"]).iterdir()], f'test_df.csv not found!'
        test_ae_df = pd.read_csv(cur_dir / config["dataset_dir"] / 'test_ae_df.csv')
        test_ae_df = test_ae_df[:3]
        model = models[args.model]

        start_time = datetime.now()
        test_ae_df['loss'] = test_ae_df.apply(lambda x: make_ae_predictions(model, x.X_test, x.y_test), axis=1)
        time_elapsed = datetime.now() - start_time 
        print(f'Predicted {test_ae_df.shape[0]} with {model.name} in {time_elapsed}(hh:mm:ss.ms).')

        test_ae_df.to_csv(cur_dir / config["dataset_dir"] / f'{args.model}_prediction.csv', index=False)

    elif (args.model == 'myconv2d'):
        # load the test set
        assert 'test_df.csv' in [x.name for x in (cur_dir / config["dataset_dir"]).iterdir()], f'test_df.csv not found!'
        test_df = pd.read_csv(cur_dir / config["dataset_dir"] / 'test_df.csv')

        le = LabelEncoder()
        encoded = le.fit_transform(test_df['y_test'])

        model = models[args.model]

        start_time = datetime.now()
        test_df['pred'] = test_df.apply(lambda x: make_predictions(model, le, x.X_test, x.y_test), axis=1)
        time_elapsed = datetime.now() - start_time 
        print(f'Predicted {test_df.shape[0]} with {model.name} in {time_elapsed}(hh:mm:ss.ms).')

        test_df.to_csv(cur_dir / config["dataset_dir"] / f'{args.model}_prediction.csv', index=False)

    device.reset()    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Malfunctioning Industrial Machine Investigation and Inspection (MIMII). Classification/Anomaly Detection Inference.')
    parser.add_argument('--model', type=str, default='myconv2dae',
                        help='model to train. (myconv1d, myconv2d, mylstm, myconv1dae, myconv2dae')
    args, _ = parser.parse_known_args()

    infer(args)