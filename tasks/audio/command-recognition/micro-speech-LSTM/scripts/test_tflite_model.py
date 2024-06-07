#!/usr/bin/env python

# SPDX-License-Identifier: MIT
# Copyright 2023-2024 NXP

import numpy as np
import tensorflow as tf
import argparse

import dataset


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", action="store", help="TFLite Model to test")
    args = argparser.parse_args()

    print("TensorFlow Version: {}".format(tf.version.VERSION))
    print(f"Loading TensorFlow Lite model {args.model}")
    dataset.seed_tf_random()

    tflite_interpreter = tf.lite.Interpreter(model_path=args.model)
    # Get input and output details
    input_details = tflite_interpreter.get_input_details()
    output_details = tflite_interpreter.get_output_details()
    tflite_interpreter.allocate_tensors()

    input_scale, input_zero_point = input_details[0]["quantization"]
    is_input_quantized = input_details[0]["dtype"] != np.float32
    input_dtype = input_details[0]["dtype"]
    print(f"Model has quantized input: {is_input_quantized}")
    print(
        f"Model's input quantization parameters: scale = {input_scale}, zero_point = {input_zero_point}"
    )

    _, _, test_ds = dataset.get_datasets(data_dir=dataset.data_dir)

    correct_class = 0
    for spectrogram, label in test_ds.batch(1):
        spectrogram = np.array(spectrogram)
        if is_input_quantized:
            spectrogram = dataset.quantize(
                spectrogram, input_scale, input_zero_point, input_dtype
            )

        spectrogram = np.array(spectrogram, dtype=input_details[0]["dtype"]).reshape(
            1, 49, 257
        )

        tflite_interpreter.set_tensor(input_details[0]["index"], spectrogram)
        tflite_interpreter.invoke()
        tflite_model_predictions = tflite_interpreter.get_tensor(
            output_details[0]["index"]
        )
        tflite_label = np.argmax(tflite_model_predictions)
        if tflite_label == label:
            correct_class += 1

    print("Accuracy: {}".format(correct_class / len(test_ds)))
