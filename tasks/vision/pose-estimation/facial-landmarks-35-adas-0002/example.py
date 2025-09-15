#!/usr/bin/env python3
# Copyright 2024 NXP
# SPDX-License-Identifier: MIT

import cv2
import tensorflow as tf
import numpy as np
import time
import random
from utils import load_image

random.seed(1337)

OBJECT_DETECTOR_TFLITE = "model_full_integer_quant.tflite"
IMAGE_FILENAME = "face.jpg"
OUTPUT_FILENAME = "example_output.jpg"

interpreter = tf.lite.Interpreter(OBJECT_DETECTOR_TFLITE)
interpreter.allocate_tensors()


def run_inference(interpreter, image):

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_scale, input_zero_point = input_details[0]["quantization"]

    image = image / input_scale + input_zero_point
    image = image.astype(np.uint8)

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()

    out_scale, out_zero_point = output_details[0]["quantization"]

    out = interpreter.get_tensor(output_details[0]['index']).astype(np.float32)

    out = (out - out_zero_point) * out_scale

    return out


if __name__ == "__main__":

    orig_image, processed_image = load_image(IMAGE_FILENAME)

    start = time.time()
    points = run_inference(interpreter, processed_image)
    end = time.time()

    # # rescale points for display
    shp = orig_image.shape
    points = points.reshape((-1, 2))
    points *= np.array([shp[1], shp[0]])
    points = points.astype(np.int32)

    print("Inference time", end - start, "ms")

    for i in range(points.shape[0]):
        point = points[i, ...]
        cv2.circle(orig_image, (point[0], point[1]), 2, (255, 0, 0), -1)

    cv2.imwrite(OUTPUT_FILENAME, orig_image)
