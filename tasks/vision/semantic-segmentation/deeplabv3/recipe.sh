#!/usr/bin/env bash
# Copyright 2023-2025 NXP
# SPDX-License-Identifier: MIT

set -e

wget --no-check-certificate https://tfhub.dev/sayakpaul/lite-model/deeplabv3-mobilenetv2_dm05-int8/1/default/2?lite-format=tflite -O deeplabv3.tflite
