#!/usr/bin/env bash

# SPDX-License-Identifier: MIT
# Copyright 2023 NXP

set -e

wget https://tfhub.dev/sayakpaul/lite-model/deeplabv3-mobilenetv2_dm05-int8/1/default/2?lite-format=tflite -O deeplabv3.tflite