#!/usr/bin/env bash
# Copyright 2024 NXP
# SPDX-License-Identifier: MIT

set -e

export MODEL=efficientnet-lite0
wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/lite/${MODEL}.tar.gz
tar zxf ${MODEL}.tar.gz
wget https://upload.wikimedia.org/wikipedia/commons/f/fe/Giant_Panda_in_Beijing_Zoo_1.JPG -O example_input.jpg

mv ${MODEL}/${MODEL}-int8.tflite ./

rm ${MODEL}.tar.gz
rm -rf ${MODEL}