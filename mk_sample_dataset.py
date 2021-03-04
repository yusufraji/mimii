from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import yaml
from tqdm import tqdm

np.random.seed(42)


# reads the audio from file, and write it to the sample_dataset folder
# to create the write the sample dataset to file
def create_dataset(file_path, dataset_dir):
    """loads the audio (sample set) from full dataset directory and writes them to sample dataset directory

    Args:
        file_path (path): sample dataset directory
        dataset_dir (path): full dataset directory
    """
    y, sr = librosa.load(file_path, sr=None)
    out_path = dataset_dir / Path(*list(file_path.parts[-5:]))
    Path.mkdir(out_path.parent, parents=True, exist_ok=True)
    sf.write(str(out_path), y, sr)


# load config yaml
with open("config.yaml") as stream:
    config = yaml.safe_load(stream)

print(
    "================================= creating folders =================================\n"
)
# create folders
Path.mkdir(Path.cwd() / config["sample_dataset_dir"], exist_ok=True)
Path.mkdir(
    Path.cwd() / config["sample_dataset_dir"] / "min6db", parents=True, exist_ok=True
)
Path.mkdir(
    Path.cwd() / config["sample_dataset_dir"] / "0db", parents=True, exist_ok=True
)
Path.mkdir(
    Path.cwd() / config["sample_dataset_dir"] / "6db", parents=True, exist_ok=True
)

# collect and save sample dataset
# write the machine type, id, and file path to dataframe
# fetch dataset
print(
    "================================= fetching dataset =================================\n"
)
data = pd.DataFrame(columns=["machine_type", "machine_id", "normal", "abnormal", "db"])
for db in tqdm(["0dB", "6dB", "min6dB"], desc="fetching db files.", leave=False):
    for folder in tqdm(
        (Path.cwd() / config["dataset_dir"] / db).iterdir(),
        desc=f"fetching machine files.",
        leave=False,
    ):
        for id in tqdm(
            folder.iterdir(), desc=f"fetching machine id files.", leave=False
        ):
            normal_glob = (id / "normal").glob("*.wav")
            abnormal_glob = (id / "abnormal").glob("*.wav")
            for normal, abnormal in zip(normal_glob, abnormal_glob):
                values = [folder.name, id.name, normal, abnormal, db]
                a_dict = dict(zip(list(data.columns), values))
                data = data.append(a_dict, ignore_index=True)

print(
    "============================== writing sample datasets ==============================\n"
)
np.random.seed(42)

db = ["0dB", "6dB", "min6dB"]
machine_type = list(data.machine_type.unique())
machine_id = list(data.machine_id.unique())
for db in tqdm(db, desc="writing db files.", leave=False):
    for machine in tqdm(machine_type, desc="writing machine files.", leave=False):
        for id in tqdm(machine_id, desc="writing machine id files.", leave=False):
            n = np.random.randint(15, 50)  # to make the data imbalance
            sample_data = data.loc[data["db"] == db]
            sample_data = sample_data.loc[sample_data["machine_type"] == machine]
            sample_data = sample_data.loc[sample_data["machine_id"] == id]
            sample_data = sample_data.sample(n=n, random_state=42)
            sample_data.reset_index(drop=True, inplace=True)

            # write dataset for the normal data
            sample_data["normal"].apply(
                create_dataset, dataset_dir=config["sample_dataset_dir"]
            )
            # write dataset for the abnormal data
            sample_data["abnormal"].apply(
                create_dataset, dataset_dir=config["sample_dataset_dir"]
            )
