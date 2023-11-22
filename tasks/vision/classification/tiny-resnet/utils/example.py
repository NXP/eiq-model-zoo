#!/usr/bin/env python3
# Copyright 2023 NXP
# SPDX-License-Identifier: MIT

import argparse

import cv2
import numpy as np
import tensorflow as tf


CIFAR10_LABELS = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}


def inference(modelPath, inputPath, isQuantized=False):
    tensorType = np.int8 if isQuantized else np.float32

    img = cv2.imread(inputPath, 1)
    img = cv2.resize(img, (32, 32))
    if isQuantized:
        img = img.astype(np.int64) - 128
    img = img.astype(tensorType)

    img = np.expand_dims(img, axis=0)

    interpreter = tf.lite.Interpreter(model_path=modelPath)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()

    pred_y = interpreter.get_tensor(output_details[0]["index"])
    if isQuantized:
        pred_y_quant = output_details[0]["quantization"]
        pred_y = pred_y.astype(np.float32)
        pred_y = pred_y_quant[0] * (pred_y - pred_y_quant[1])

    print(CIFAR10_LABELS[pred_y.argmax()])


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-i",
        "--image",
        action="store",
        dest="inputSamplePath",
        default="example_input.jpg",
        help="Path to input image.",
    )
    argparser.add_argument(
        "-m",
        "--model",
        action="store",
        dest="modelPath",
        default="pretrainedResnet.tflite",
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
