#!/usr/bin/env bash
# Copyright 2023 NXP

set -e

# Model scale (i.e. model size) can be 0.25, 0.50, 0.75, 1.0
MODEL_SCALE=0.25

# Model input resolution can be 128, 160, 192, 224
MODEL_INPUT_RESOLUTION=128

ARCHIVE_NAME=mobilenet_v1_${MODEL_SCALE}_${MODEL_INPUT_RESOLUTION}_quant.tgz

## Download pre-trained model and extract to tmp dir
mkdir -p tmp
wget http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/${ARCHIVE_NAME}
tar xzvf ${ARCHIVE_NAME} -C tmp
rm ${ARCHIVE_NAME}

cp  tmp/mobilenet_v1_${MODEL_SCALE}_${MODEL_INPUT_RESOLUTION}_quant.tflite .

rm -rf tmp

## install vela and convert model for i.MX 93
python3 -m venv env
source ./env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

pip install git+https://github.com/nxp-imx/ethos-u-vela.git@lf-6.1.1-1.0.0

vela --output-dir model_imx93 mobilenet_v1_${MODEL_SCALE}_${MODEL_INPUT_RESOLUTION}_quant.tflite

deactivate

rm -rf env