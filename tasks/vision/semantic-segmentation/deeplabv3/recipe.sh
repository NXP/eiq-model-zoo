#!/usr/bin/env bash
# Copyright 2023 NXP
# SPDX-License-Identifier: MIT

set -e

wget https://tfhub.dev/sayakpaul/lite-model/deeplabv3-mobilenetv2_dm05-int8/1/default/2?lite-format=tflite -O deeplabv3.tflite
