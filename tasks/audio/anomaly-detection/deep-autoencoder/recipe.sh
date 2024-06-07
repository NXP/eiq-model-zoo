#!/bin/sh

# SPDX-License-Identifier: MIT
# Copyright 2023-2024 NXP
set -e

python3 -m venv env
source ./env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

wget https://github.com/mlcommons/tiny/raw/master/benchmark/training/anomaly_detection/trained_models/ad01_int8.tflite

deactivate
rm -rf env
