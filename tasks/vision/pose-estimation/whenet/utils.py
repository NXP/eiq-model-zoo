#!/usr/bin/env python3

# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2020. Huawei Technologies Co., Ltd.
#  All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from this
# software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
from math import cos, sin
import cv2
import efficientnet.keras as efn
import keras


# From https://github.com/revygabor/WHENet/blob/2986edad2a02557ab4b4849938fbcc5549a8f3eb/utils.py
def softmax(x):
    return np.exp(x) / np.exp(x).sum()


# From https://github.com/revygabor/WHENet/blob/2986edad2a02557ab4b4849938fbcc5549a8f3eb/utils.py
def draw_axis(img, yaw, pitch, roll):
    # Referenced from HopeNet https://github.com/natanielruiz/deep-head-pose
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180
    size = 100

    height, width = img.shape[:2]
    tdx = width / 2
    tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll)
                 + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll)
                 - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 2)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 2)
    return img


# From https://github.com/revygabor/WHENet/blob/2986edad2a02557ab4b4849938fbcc5549a8f3eb/whenet.py
def normalize(im):

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    im = im[None, ...]/255
    im = (im - mean) / std

    return im


# From https://github.com/revygabor/WHENet/blob/2986edad2a02557ab4b4849938fbcc5549a8f3eb/whenet.py
def decode(y, p, r):

    idx_tensor = [idx for idx in range(66)]
    idx_tensor = np.array(idx_tensor, dtype=np.float32)

    idx_tensor_yaw = [idx for idx in range(120)]
    idx_tensor_yaw = np.array(idx_tensor_yaw, dtype=np.float32)

    # Output decoding according to https://arxiv.org/pdf/2005.10353.pdf eq. 2
    yaw_predicted = softmax(y)
    pitch_predicted = softmax(p)
    roll_predicted = softmax(r)
    yaw_predicted = np.sum(yaw_predicted * idx_tensor_yaw, axis=1) * 3-180
    pitch_predicted = np.sum(pitch_predicted * idx_tensor, axis=1) * 3 - 99
    roll_predicted = np.sum(roll_predicted * idx_tensor, axis=1) * 3 - 99

    return yaw_predicted, pitch_predicted, roll_predicted


# From https://github.com/revygabor/WHENet/blob/2986edad2a02557ab4b4849938fbcc5549a8f3eb/whenet.py
def get_whenet():

    base_model = efn.EfficientNetB0(include_top=False,
                                    input_shape=(224, 224, 3))
    out = base_model.output
    out = keras.layers.GlobalAveragePooling2D()(out)
    fc_yaw = keras.layers.Dense(name='yaw_new',
                                units=120)(out)  # 3 * 120 = 360 degrees in yaw
    fc_pitch = keras.layers.Dense(name='pitch_new', units=66)(out)
    fc_roll = keras.layers.Dense(name='roll_new', units=66)(out)
    model = keras.models.Model(inputs=base_model.input,
                               outputs=[fc_yaw, fc_pitch, fc_roll])

    return model
