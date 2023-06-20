#!/usr/bin/env bash

# SPDX-License-Identifier: MIT
# Copyright 2023 NXP

set -e

wget https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/int8/4?lite-format=tflite -O movenet.tflite

# Install requirements in virtual env
python3 -m venv env
source ./env/bin/activate
pip install --upgrade pip
pip install numpy==1.23.5
# install vela
pip install git+https://github.com/nxp-imx/ethos-u-vela.git@lf-6.1.1-1.0.0

vela --output-dir model_imx93 movenet.tflite

deactivate

rm -rf env