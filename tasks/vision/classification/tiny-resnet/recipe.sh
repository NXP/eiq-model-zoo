#!/usr/bin/env bash
# Copyright 2023-2025 NXP
# SPDX-License-Identifier: MIT

python3 -m venv env
source ./env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

wget https://github.com/mlcommons/tiny/raw/master/benchmark/training/image_classification/trained_models/pretrainedResnet.h5
wget https://github.com/mlcommons/tiny/raw/master/benchmark/training/image_classification/trained_models/pretrainedResnet.tflite
wget https://github.com/mlcommons/tiny/raw/master/benchmark/training/image_classification/trained_models/pretrainedResnet_quant.tflite


pip install git+https://github.com/nxp-imx/ethos-u-vela.git@lf-6.1.22-2.0.0

vela --output-dir model_imx93 pretrainedResnet_quant.tflite
deactivate

rm -rf env
