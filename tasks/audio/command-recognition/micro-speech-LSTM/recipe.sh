#!/bin/sh

# SPDX-License-Identifier: MIT
# Copyright 2023-2024 NXP

set -e

python3.8 -m venv env
source ./env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

if [ ! -d "model" ]; then
  mkdir model
fi

python3.8 ./scripts/train_model.py
python3.8 ./scripts/convert_model.py

deactivate
rm -rf env
