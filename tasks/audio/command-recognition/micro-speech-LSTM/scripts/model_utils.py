#!/usr/bin/env python

# SPDX-License-Identifier: MIT
# Copyright 2023-2024 NXP

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models


# Set seed for experiment reproducibility

MODEL_DIR = "model"
MODEL_NAME = "micro_speech_with_lstm_op.keras"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)


def get_model(unroll=False):
    model = models.Sequential(
        [
            layers.Input(shape=(49, 257), name="input"),
            layers.Reshape(target_shape=(49, 257)),
            layers.LSTM(80, time_major=False, return_sequences=True, unroll=unroll),
            layers.Flatten(),
            layers.Dense(3, activation=tf.nn.softmax, name="output"),
        ]
    )
    return model
