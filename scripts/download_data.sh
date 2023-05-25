#!/bin/bash

# Create the data directory
mkdir -p "data"

# Download data from kaggle
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
kaggle datasets download -d paultimothymooney/kermany2018

# =========================== Chest X-Ray Images ============================ #

unzip -q "chest-xray-pneumonia.zip" -d "data"
rm -r "chest-xray-pneumonia.zip" "data/chest_xray/__MACOSX" "data/chest_xray/chest_xray"

# =========================== Kermany2018 =================================== #

unzip -q "kermany2018.zip" -d "data/oct"
mv data/oct/OCT2017\ /t* data/oct && mv data/oct/OCT2017\ /v* data/oct
rmdir data/oct/OCT2017\ /
rm -r "data/oct/oct2017" "kermany2018.zip"
