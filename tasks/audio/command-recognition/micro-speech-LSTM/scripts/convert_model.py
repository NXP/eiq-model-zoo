#!/usr/bin/env python

# SPDX-License-Identifier: MIT
# Copyright 2023-2024 NXP

import os

import numpy as np
import tensorflow as tf

import dataset
import model_utils

MODEL_NAME = 'microspeech-lstm'


def representative_dataset():
    train_ds, _, _ = dataset.get_datasets(data_dir=dataset.data_dir)
    for spectrogram, _ in train_ds.take(800):
        flattened_data = np.array(spectrogram, dtype=np.float32).reshape(1, 49, 257)
        yield [flattened_data]


def convert_to_tflite(model_path):
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.representative_dataset = representative_dataset

    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    tflite_path = os.path.join(MODEL_NAME + "_quant_int8.tflite")

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"Model stored to {tflite_path}")


if __name__ == "__main__":
    print("TensorFlow Version: {}".format(tf.version.VERSION))
    print(f"Loading TensorFlow model from {model_utils.MODEL_PATH}")
    dataset.seed_tf_random()
    model = tf.keras.models.load_model(model_utils.MODEL_PATH)

    run_model = tf.function(lambda x: model(x))
    # This is important, let's fix the input size.
    BATCH_SIZE = 1
    STEPS = 49
    INPUT_SIZE = 257
    concrete_func = run_model.get_concrete_function(
        tf.TensorSpec([BATCH_SIZE, STEPS, INPUT_SIZE], model.inputs[0].dtype)
    )

    model.save(model_utils.MODEL_DIR, save_format="tf", signatures=concrete_func)

    print(
        "============== Converting to int8 tflite model ============================="
    )
    convert_to_tflite(model_utils.MODEL_DIR)
