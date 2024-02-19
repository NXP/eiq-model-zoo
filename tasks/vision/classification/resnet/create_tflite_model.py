#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright 2023-2024 NXP

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
from tqdm import tqdm

h, w = 224, 224

dataset, ds_info = tfds.load('imagenet_v2/topimages', split='test', with_info=True)


def dataset_gen():
    for item in tqdm(dataset.take(500)):
        img = np.array(item['image'])
        img = Image.fromarray(img)
        if img.mode != "RGB":
            continue
        img = img.resize((h, w))
        img_array = np.array(img)
        tensor = tf.keras.applications.resnet_v2.preprocess_input(img_array)
        tensor = np.expand_dims(tensor, axis=0)
        yield [tensor]


model = tf.keras.applications.resnet_v2.ResNet50V2(
    include_top=True,
    weights='imagenet'
)

# print(ds_info)

"""
Convert to tflite float
"""
converter = tf.lite.TFLiteConverter.from_keras_model(model)
model_tf = converter.convert()
with open("resnet50_fp32.tflite", "wb") as f:
    f.write(model_tf)

"""
Int8 quantization
"""
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    tf.lite.OpsSet.TFLITE_BUILTINS]
converter.representative_dataset = dataset_gen
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_quant_int8_model = converter.convert()
with open("resnet50_int8.tflite", "wb") as f:
    f.write(tflite_quant_int8_model)
