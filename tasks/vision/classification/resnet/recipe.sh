#!/bin/sh

# SPDX-License-Identifier: MIT
# Copyright 2023-2024 NXP

set -e

python3 -m venv env
source ./env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

python3 create_tflite_model.py

deactivate
rm -rf env