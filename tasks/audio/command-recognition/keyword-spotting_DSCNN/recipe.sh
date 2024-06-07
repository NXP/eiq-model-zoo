#!/bin/sh

# SPDX-License-Identifier: MIT
# Copyright 2023-2024 NXP

set -e

python3.8 -m venv env
source ./env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

wget https://github.com/mlcommons/tiny/raw/master/benchmark/training/keyword_spotting/trained_models/kws_ref_model.tflite
wget https://github.com/mlcommons/tiny/raw/master/benchmark/training/keyword_spotting/trained_models/kws_ref_model_float32.tflite


deactivate
rm -rf env
