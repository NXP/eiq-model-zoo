#!/bin/sh

# SPDX-License-Identifier: MIT
# Copyright 2024 NXP

set -e

python3 -m venv env
source ./env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

curl -L -o model.tar.gz\
  https://www.kaggle.com/api/v1/models/tensorflow/efficientdet/tfLite/lite0-int8/1/download

tar -xvf model.tar.gz
mv 1.tflite efficientdet-lite0_quant_int8.tflite

wget https://raw.githubusercontent.com/amikelive/coco-labels/master/coco-labels-2014_2017.txt
