#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright 2023 NXP

import tensorflow as tf
import numpy as np
from tqdm import tqdm


h, w = 224, 224
dataset = np.load('quantization_data.npy')

def dataset_gen():
    for item in tqdm(dataset):
        item = np.expand_dims(item, axis=0)
        yield [item]


model = tf.keras.applications.MobileNetV2(input_shape=(h, w, 3))

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.input_shape = (1, h, w, 3)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = dataset_gen
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    tf.lite.OpsSet.TFLITE_BUILTINS,
]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_quant_model = converter.convert()

with open("mobilenet_v2_quant_int8.tflite", "wb") as f:
    f.write(tflite_quant_model)
print("Model saved.")
