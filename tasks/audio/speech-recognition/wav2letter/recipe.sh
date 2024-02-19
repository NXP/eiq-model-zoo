#!/bin/sh

# SPDX-License-Identifier: MIT
# Copyright 2023 NXP

set -e

python3.8 -m venv env
source ./env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

wget -P . https://github.com/ARM-software/ML-zoo/raw/master/models/speech_recognition/wav2letter/tflite_int8/wav2letter_int8.tflite

deactivate
rm -rf env

