from datetime import datetime
from pathlib import Path

import librosa
import numpy as np
import yaml
from librosa.feature.spectral import melspectrogram
from scipy.io import wavfile
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

start_time = datetime.now()

# load config yaml
with open("config.yaml") as stream:
    config = yaml.safe_load(stream)

# generate a batch of time data. X = Y = [32,128,313,8]
# reference:
# https://stackoverflow.com/questions/51241499/parameters-to-control-the-size-of-a-spectrogram
n_frames = int(
    np.floor(
        (config["feature"]["sr"] * config["feature"]["dt"])
        / config["feature"]["hop_length"]
    )
    + 1
)
# n_frames must be even for autoencoder symmetry
# round n_frames up to the nearest even value!
n_frames = int(np.ceil(n_frames / 2.0) * 2)

# 1. read audio dataset.
# 2. convert to melspectrogram.
# 3. write melspectrogram as numpy array to file in the same directory as the
#    wav file.
print(
    "============================== preprocessing dataset ==============================\n"
)
wav_files = (Path() / config["dataset_dir"]).glob("**/*.wav")
for item in tqdm(wav_files, desc="preprocessing wav files.", leave=False):
    rate, wav = wavfile.read(item)
    # change the wav file format to float since wavfile.read returns int
    # and librosa.feature.melspectrogram expects float wav = [160000,8]
    wav = wav.astype(np.float)

    # convert the audio to melspectrogram one channel at a time. since
    # librosa only works on mono S_dB = [128,313,8]
    # reference: https://stackoverflow.com/questions/51241499/parameters-to-control-the-size-of-a-spectrogram
    S_db = np.empty(
        (config["feature"]["n_mels"], n_frames, config["feature"]["n_channels"]),
        dtype=np.float32,
    )
    for j in range(config["feature"]["n_channels"]):
        S = librosa.feature.melspectrogram(
            y=wav[:, j],
            sr=config["feature"]["sr"],
            hop_length=config["feature"]["hop_length"],
        )
        # amplitude_to_db and pad with zeros to the length of even
        # adjusted n_frames
        S_db_tmp = librosa.amplitude_to_db(S, ref=np.max)
        # reference: https://stackoverflow.com/questions/38191855/zero-pad-numpy-array/38192105
        S_db[:, :, j] = np.pad(
            S_db_tmp, ((0, 0), (0, n_frames - S_db_tmp.shape[1])), "constant"
        )
        # normalize
        minmax = MinMaxScaler()
        S_db[:, :, j] = minmax.fit_transform(S_db[:, :, j])

    # save the converted melspectrogram numpy array to file
    np.save((item.parent / item.stem), S_db)

time_elapsed = datetime.now() - start_time
print(f"preprocessing elapsed (hh:mm:ss.ms) {time_elapsed}")
# preprocessing elapsed (hh:mm:ss.ms) 2:44:14.563688
