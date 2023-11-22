#!/bin/sh

# SPDX-License-Identifier: MIT
# Copyright 2023 NXP

set -e

python3 -m venv env
source ./env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

python3 utils/generate_data.py
python3 utils/create_tflite_models.py

deactivate

rm -rf quantization_data.npy
rm -rf env
