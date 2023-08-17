#!/usr/bin/env bash

# SPDX-License-Identifier: MIT
# Copyright 2023 NXP

set -e

wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights
wget https://raw.githubusercontent.com/amikelive/coco-labels/master/coco-labels-2014_2017.txt


# tensorflow -> tflite
python3.8 -m venv env
source ./env/bin/activate

pip install --upgrade pip
pip install tensorflow==2.10.0
pip install Pillow

wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip

# convert model from darknet to tensorflow lite
python3.8 export_model.py --weights_path=./ --output_path=./ --images_path=val2017

# install vela
pip install numpy==1.20
pip install git+https://github.com/nxp-imx/ethos-u-vela.git@lf-6.1.22-2.0.0

vela --output-dir model_imx93 yolov4-tiny_416_quant.tflite

# cleanup 
deactivate
rm -rf val2017 env
rm val2017.zip
rm yolov4-tiny.weights
