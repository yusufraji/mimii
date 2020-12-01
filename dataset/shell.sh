
mkdir ./min6dB
mkdir ./6dB
mkdir ./0dB

wget -c -O min6dbfan.zip https://zenodo.org/record/3384388/files/-6_dB_fan.zip
unzip -q min6dbfan.zip -d ./min6dB
rm min6dbfan.zip

wget -c -O min6dbpump.zip https://zenodo.org/record/3384388/files/-6_dB_pump.zip
unzip -qq min6dbpump.zip -d ./min6dB
rm min6dbpump.zip

wget -c -O min6dbslider.zip https://zenodo.org/record/3384388/files/-6_dB_slider.zip
unzip -qq min6dbslider -d ./min6dB
rm min6dbslider.zip

wget -c -O min6dbvalve.zip https://zenodo.org/record/3384388/files/-6_dB_valve.zip
unzip -qq min6dbvalve -d ./min6dB
rm min6dbvalve.zip

wget -c -O 6dbfan.zip https://zenodo.org/record/3384388/files/6_dB_fan.zip
unzip -qq 6dbfan.zip -d ./6dB
rm 6dbfan.zip

wget -c -O 6dbpump.zip https://zenodo.org/record/3384388/files/6_dB_pump.zip
unzip -qq 6dbpump.zip -d ./6dB
rm 6dbpump.zip

wget -O 6dbslider.zip https://zenodo.org/record/3384388/files/6_dB_slider.zip
unzip -qq 6dbslider -d ./6dB
rm 6dbslider.zip

wget -c -O 6dbvalve.zip https://zenodo.org/record/3384388/files/6_dB_valve.zip
unzip -qq 6dbvalve -d ./6dB
rm 6dbvalve.zip

wget -c -O 0dbfan.zip https://zenodo.org/record/3384388/files/0_dB_fan.zip
unzip -qq 0dbfan.zip -d ./0dB
rm 0dbfan.zip

wget -c -O 0dbpump.zip https://zenodo.org/record/3384388/files/0_dB_pump.zip
unzip -qq 0dbpump.zip -d ./0dB
rm 0dbpump.zip

wget -c -O 0dbslider.zip https://zenodo.org/record/3384388/files/0_dB_slider.zip
unzip -qq 0dbslider -d ./0dB
rm 0dbslider.zip

wget -c -O 0dbvalve.zip https://zenodo.org/record/3384388/files/0_dB_valve.zip
unzip -qq 0dbvalve -d ./0dB
rm 0dbvalve.zip