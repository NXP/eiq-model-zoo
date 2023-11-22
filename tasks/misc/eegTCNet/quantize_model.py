#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright 2023 NXP

import importlib
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

data_loading = importlib.import_module("eeg-tcnet-master.utils.data_loading")

DATA_PATH = os.path.join(os.getcwd(), "data/")

MODEL_PATH = os.path.join(
    os.getcwd(), "eeg-tcnet-master/models/EEG-TCNet/S1/model_fixed.h5"
)
keras_model = keras.models.load_model(MODEL_PATH)

converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    tf.lite.OpsSet.SELECT_TF_OPS,
]

X_train, _, y_train_onehot, X_test, _, y_test_onehot = data_loading.prepare_features(
    DATA_PATH, 0, False
)

X_train = X_train.astype(np.float32)


def dataset_gen():
    for row in tqdm(X_train):
        row = np.expand_dims(row, axis=0)
        yield [row]


converter.representative_dataset = dataset_gen

converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()
open("eegTCNet_quant_int8.tflite", "wb").write(tflite_model)
