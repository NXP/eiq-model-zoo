#!/usr/bin/env bash
# Copyright 2024 NXP
# SPDX-License-Identifier: MIT

set -e

python3.8 -m venv env
source ./env/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

curl "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/171_Fast-SRGAN/resources.tar.gz" -o resources.tar.gz
tar -zxvf resources.tar.gz
rm resources.tar.gz

mv model_128x128/model_integer_quant.tflite fsrgan.tflite

# cleanup
rm -rf model_*
deactivate
