#!/usr/bin/env python

# SPDX-License-Identifier: MIT
# Copyright 2023-2024 NXP

import numpy as np
import tensorflow as tf
import argparse

import dataset

LABELS = ["no", "unknown", "yes"]

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--model",
        action="store",
        default="model/model_int8.tflite",
        help="TFLite Model to test",
    )
    argparser.add_argument(
        "--data", action="store", default="example.wav", help="Input data for inference"
    )
    args = argparser.parse_args()

    tflite_interpreter = tf.lite.Interpreter(model_path=args.model)
    # Get input and output details
    input_details = tflite_interpreter.get_input_details()
    output_details = tflite_interpreter.get_output_details()
    tflite_interpreter.allocate_tensors()
    input_scale, input_zero_point = input_details[0]["quantization"]
    is_input_quantized = input_details[0]["dtype"] != np.float32
    input_dtype = input_details[0]["dtype"]

    audio_binary = tf.io.read_file(args.data)
    waveform = dataset.decode_audio(audio_binary)
    spectrogram = dataset.get_spectrogram(waveform)
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
    tflite_model_predictions = tflite_interpreter.get_tensor(output_details[0]["index"])
    tflite_label = np.argmax(tflite_model_predictions)
    print("Recognized word: ", LABELS[tflite_label])
