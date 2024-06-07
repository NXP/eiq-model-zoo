#!/usr/bin/env bash

# SPDX-License-Identifier: MIT
# Copyright 2023-2024 NXP

set -e

python3 -m venv env
source ./env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt


mkdir -p keras_model
cd keras_model || exit
wget https://storage.googleapis.com/cloud-tpu-checkpoints/mnasnet/mnasnet-a1-075.tgz -O mnasnet-a1-075.tgz
tar xzf mnasnet-a1-075.tgz
rm mnasnet-a1-075.tgz

cd ..
python3 ./utils/prepare_data.py
python3 ./utils/create_tflite_models.py


deactivate
rm -rf quantization_data.npy
rm -rf env
