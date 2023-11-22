#!/usr/bin/env python

# SPDX-License-Identifier: MIT
# Copyright 2023 NXP

import os
import numpy as np
import tensorflow as tf

import model_utils
import dataset


def representative_dataset_3():
    train_ds, _, _ = dataset.get_datasets(data_dir=dataset.data_dir)
    for spectrogram, _ in train_ds.take(800):
        # print('test')
        flattened_data = np.array(spectrogram, dtype=np.float32).reshape(1, 49, 257)
        yield [flattened_data]


def convert_to_tflite_float(model_path, output_path, output_name="model_float"):
    # Float LSTM model
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    tflite_float_model = converter.convert()
    open(os.path.join(output_path, output_name + ".tflite"), "wb").write(
        tflite_float_model
    )


def convert_to_tflite_int8(model_path, output_path, output_name="model_int8"):
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.representative_dataset = representative_dataset_3
    tflite_int8_model = converter.convert()
    open(os.path.join(output_path, output_name + ".tflite"), "wb").write(
        tflite_int8_model
    )


def convert_to_tflite(
    model_path, output_path, type, output_name="model", quantize_inout=False
):
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.representative_dataset = representative_dataset_3
    if type == "float":
        output_name += "_float"
    elif type == "int8":
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.representative_dataset = representative_dataset_3
        if quantize_inout:
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        output_name += "_int8"
    elif type == "int16":
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
        ]
        converter.representative_dataset = representative_dataset_3
        if quantize_inout:
            converter.inference_input_type = tf.int16
            converter.inference_output_type = tf.int16
        output_name += "_int16"
    try:
        tflite_model = converter.convert()
        tflite_path = os.path.join(output_path, output_name + ".tflite")
        open(tflite_path, "wb").write(tflite_model)
        print(f"Model stored to {tflite_path}")
    except Exception as e:
        print("Warning: {}".format(e))


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
        "============== Converting to float tflite model ============================"
    )
    convert_to_tflite(model_utils.MODEL_DIR, model_utils.MODEL_DIR, "float")
    print(
        "============== Converting to int8 tflite model ============================="
    )
    convert_to_tflite(
        model_utils.MODEL_DIR, model_utils.MODEL_DIR, "int8", quantize_inout=True
    )
    print(
        "============== Converting to int16 tflite model ============================"
    )
    convert_to_tflite(
        model_utils.MODEL_DIR, model_utils.MODEL_DIR, "int16", quantize_inout=False
    )
