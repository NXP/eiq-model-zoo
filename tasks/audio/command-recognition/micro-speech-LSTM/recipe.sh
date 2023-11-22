#!/bin/sh

# SPDX-License-Identifier: MIT
# Copyright 2023 NXP

set -e

python3.8 -m venv env
source ./env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

mkdir model

python3.8 ./scripts/train_model.py
python3.8 ./scripts/convert_model.py

deactivate
rm -rf env
