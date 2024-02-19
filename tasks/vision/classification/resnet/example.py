#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright 2023-2024 NXP

import argparse

import numpy as np
import tensorflow as tf
from PIL import Image

h, w = 224, 224


def quantize(input, scale, zp):
    return (input / scale) + zp


def inference(modelPath, img, isQuantized=True):
    interpreter = tf.lite.Interpreter(model_path=modelPath)
    input_details = interpreter.get_input_details()

    img = Image.open(img)
    img = img.resize((224, 224))
    img = np.array(img)
    img = tf.keras.applications.resnet_v2.preprocess_input(img)

    if isQuantized:
        scale, zero_point = input_details[0]['quantization']
        img = quantize(img, scale, zero_point)
        img = img.astype(np.int8)

    tensor = np.expand_dims(img, axis=0)

    if tensor.shape[-1] != 3:
        return None

    output_details = interpreter.get_output_details()
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]["index"], tensor)
    interpreter.invoke()

    pred_y = interpreter.get_tensor(output_details[0]["index"])

    if isQuantized:
        pred_y_quant = output_details[0]["quantization"]
        pred_y = pred_y.astype(np.float32)
        pred_y = pred_y_quant[0] * (pred_y - pred_y_quant[1])

    return tf.keras.applications.resnet_v2.decode_predictions(pred_y)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-m",
        "--model",
        action="store",
        dest="modelPath",
        default="resnet50_fp32.tflite",
        help="Path to TFLite model.",
    )
    argparser.add_argument(
        "-i",
        "--input",
        action="store",
        dest="inputPath",
        default="example_input.jpg",
        help="Path to input image"
    )
    argparser.add_argument(
        "-q",
        action="store_true",
        dest="modelQuantized",
        help="True if model is quantized",
    )

    args = argparser.parse_args()

    predictions = inference(args.modelPath, args.inputPath, args.modelQuantized)
    print(predictions)
