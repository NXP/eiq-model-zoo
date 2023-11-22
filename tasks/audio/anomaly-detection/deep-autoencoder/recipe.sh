#!/bin/sh

# SPDX-License-Identifier: MIT
# Copyright 2023 NXP
set -e

python3 -m venv env
source ./env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

wget https://github.com/mlcommons/tiny/raw/master/benchmark/training/anomaly_detection/trained_models/ad01_int8.tflite

if [ ! -d "original_dataset" ]; then
  mkdir -p original_dataset
  ZIPFILE="ToyCar.zip"

  curl https://zenodo.org/record/3678171/files/dev_data_ToyCar.zip?download=1 -o $ZIPFILE
  unzip $ZIPFILE -d original_dataset
  rm $ZIPFILE
fi

deactivate
rm -rf env
