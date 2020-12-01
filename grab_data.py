import os
import wget
from pathlib import Path
import subprocess

path = Path.cwd()
dataset_path = path / 'dataset'
os.chdir(dataset_path)
# download the dataset
# url = 'https://raw.githubusercontent.com/Call-for-Code/Spot-Challenge-Wildfires/main/data/Nov_10.zip'
# wget.download(url)

# url = 'https://zenodo.org/record/3384388/files/-6_dB_fan.zip'
# wget.download(url, 'min6dbfan.zip')
# url = 'https://zenodo.org/record/3384388/files/-6_dB_pump.zip'
# wget.download(url, 'min6dbpump.zip')
# url = 'https://zenodo.org/record/3384388/files/-6_dB_slider.zip'
# wget.download(url, 'min6dbslider.zip')
# url = 'https://zenodo.org/record/3384388/files/-6_dB_valve.zip'
# wget.download(url, 'min6dbvalve.zip')

# url = 'https://zenodo.org/record/3384388/files/6_dB_fan.zip'
# wget.download(url, '6dbfan.zip')
# url = 'https://zenodo.org/record/3384388/files/6_dB_pump.zip'
# wget.download(url, '6dbpump.zip')
# url = 'https://zenodo.org/record/3384388/files/6_dB_slider.zip'
# wget.download(url, '6dbslider.zip')
# url = 'https://zenodo.org/record/3384388/files/6_dB_valve.zip'
# wget.download(url, '6dbvalve.zip')

# url = 'https://zenodo.org/record/3384388/files/0_dB_fan.zip'
# wget.download(url, '0dbfan.zip')
# url = 'https://zenodo.org/record/3384388/files/0_dB_pump.zip'
# wget.download(url, '0dbpump.zip')
# url = 'https://zenodo.org/record/3384388/files/0_dB_slider.zip'
# wget.download(url, '0dbslider.zip')
# url = 'https://zenodo.org/record/3384388/files/0_dB_valve.zip'
# wget.download(url, '0dbvalve.zip')

# url = ''
# wget.dowload(url, '')

# run script to unzip dataset
subprocess.call(['sh', './shell.sh'])
