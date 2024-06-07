#!/usr/bin/env bash
# Copyright 2023-2024 NXP
# SPDX-License-Identifier: MIT

python3 -m venv env
source ./env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

if [ ! -d "vw_coco2014_96" ]
then
  wget https://www.silabs.com/public/files/github/machine_learning/benchmarks/datasets/vw_coco2014_96.tar.gz
  tar -xvf vw_coco2014_96.tar.gz
fi

wget https://github.com/mlcommons/tiny/raw/master/benchmark/training/visual_wake_words/trained_models/vww_96.h5
wget https://github.com/mlcommons/tiny/raw/master/benchmark/training/visual_wake_words/trained_models/vww_96_float.tflite
wget https://github.com/mlcommons/tiny/raw/master/benchmark/training/visual_wake_words/trained_models/vww_96_int8.tflite



pip install git+https://github.com/nxp-imx/ethos-u-vela.git

vela --output-dir model_imx93 vww_96_int8.tflite
deactivate

rm -rf env
