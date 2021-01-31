# Welcome to the Anomalous Sound Detection of Industrial Machines repository

This repository contains the source code for the Anomalous Sound Detection of
Industrial Machines.

This work is based on the 
## Getting Started

### 1. Create virtual environment & install python packages/dependencies

Run the following commands to install all the dependencies into a virtual
environment.

Linux/Mac Bash:
```
python3 -m venv venv

source venv/bin/activate
pip install -r requirements.txt
```

Windows Cmd:
```
python3 -m venv venv

.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download dataset

To download all the dataset (.zip files) from ZENODO, run:

`sh grab_data.sh`

After downloading, you'll find the dataset in the dataset directory, or the
directory specified in `config.yaml` for `dataset_dir`.

### 2a. Create a sample dataset (optional)

This creates a sample dataset from the entire dataset to speed up development.
Instead of experimenting on the entire dataset. Run:
`python mk_sample_dataset.py`

After downloading, you'll find the sample dataset in the sample dataset directory, or the
directory specified in `config.yaml` for `sample_dataset_dir`.

### 3. Training

Run the train.py script with classifier/anomaly_detector argument to train
a classifier or anomaly detector.

Classification
```
python train.py --model classifier
```

Anomaly detection
```
python train.py --model anomaly_detector
```

After training, you'll find the training results in the results & logs directories, or the
directories specified in `config.yaml` for `results` and `log`, respectively.

Other results, such as training, validation and testing data used can be found
in the dataset directory, or the directory specified in `config.yaml` for
`dataset`. 
## Dependencies

This source code was developed on Ubuntu 18.04.3 LTS (Bionic Beaver). 

### Software packages
- wget
- unzip
- python==3.6.9

### Python packages
Please refer to `requirements.txt` for a full list of python packages and their respective versions.

## References

> [1] Harsh Purohit, Ryo Tanabe, Kenji Ichige, Takashi Endo, Yuki Nikaido, Kaori Suefusa, and Yohei Kawaguchi, “MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection,” arXiv preprint arXiv:1909.09347, 2019. URL: https://arxiv.org/abs/1909.09347

> [2] Harsh Purohit, Ryo Tanabe, Kenji Ichige, Takashi Endo, Yuki Nikaido, Kaori Suefusa, and Yohei Kawaguchi, “MIMII Dataset: Sound for Malfunctioning Industrial Machine Investigation and Inspection,” in Proc. 4th Workshop on Detection and Classification of Acoustic Scenes and Events (DCASE), 2019.

> [3] Choi, Keunwoo, Deokjin Joo, and Juho Kim. "Kapre: On-gpu audio preprocessing layers for a quick implementation of deep neural network models with keras." arXiv preprint arXiv:1706.05781 (2017).

> [4] Dataset (https://zenodo.org/record/3384388)
