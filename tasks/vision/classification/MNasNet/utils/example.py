#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright 2023 NXP

import argparse

import numpy as np
import tensorflow as tf
from PIL import Image


def inference(modelPath, inputPath, isQuantized=True):
    img = Image.open(inputPath)
    img = img.resize((224, 224))
    if isQuantized:
        img = np.array(img, dtype=np.int64)
        img = img - 128
        tensor = img.astype(np.int8)
    else:
        tensor = np.array(img, dtype=np.float32)

    tensor = np.expand_dims(tensor, axis=0)

    interpreter = tf.lite.Interpreter(model_path=modelPath)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]["index"], tensor)
    interpreter.invoke()

    pred_y = interpreter.get_tensor(output_details[1]["index"])

    if isQuantized:
        pred_y_quant = output_details[1]["quantization"]
        pred_y = pred_y.astype(np.float32)
        pred_y = pred_y_quant[0] * (pred_y - pred_y_quant[1])

    print(tf.keras.applications.nasnet.decode_predictions(pred_y))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-i",
        "--input",
        action="store",
        dest="inputSamplePath",
        default="example_input.jpg",
        help="Path to input image",
    )
    argparser.add_argument(
        "-m",
        "--model",
        action="store",
        dest="modelPath",
        default="mnasnet-a1-075_float32.tflite",
        help="Path to TFLite model.",
    )
    argparser.add_argument(
        "-q",
        action="store_true",
        dest="modelQuantized",
        help="True if model is quantized",
    )

    args = argparser.parse_args()

    inference(args.modelPath, args.inputSamplePath, args.modelQuantized)
