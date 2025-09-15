#!/bin/sh

# SPDX-License-Identifier: MIT
# Copyright 2024 NXP

set -e

python3 -m venv env
source ./env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

curl -L -k -o model.tar.gz\
  https://www.kaggle.com/api/v1/models/tensorflow/inception/tfLite/v4-quant/1/download

tar -xvf model.tar.gz
mv 1.tflite inceptionv4_quant_int8.tflite

