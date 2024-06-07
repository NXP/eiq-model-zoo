#!/usr/bin/env python3
# Copyright 2024 NXP
# SPDX-License-Identifier: MIT

import cv2
import tensorflow as tf
import numpy as np
import time
import random
from PIL import Image

OBJECT_DETECTOR_TFLITE = 'fsrgan.tflite'
IMAGE_FILENAME = 'example_input.png'

interpreter = tf.lite.Interpreter(OBJECT_DETECTOR_TFLITE)


def load_image(filename):
    # Preprocessing to prepare the input for the model
    orig_image = cv2.imread(filename, 1)
    image = cv2.resize(orig_image, dsize=(orig_image.shape[1], orig_image.shape[0]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Add the batch dimension to the input tensor
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return orig_image, image



def run_inference(interpreter, image):

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    type_tf = output_details[0]['dtype']
    # This is used to adapt the model to every size. However, the model will still perform best on its trained input size (i.e 128x128)
    input_details[0]['shape'][1], input_details[0]['shape'][2] = image.shape[1], image.shape[2]
    interpreter.resize_tensor_input(0, input_details[0]['shape'])
    interpreter.allocate_tensors()
    image = tf.cast(image, tf.float32)
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]['index'])
    if type_tf == np.uint8:
        input_scale, input_zero_point = output_details[0]["quantization"]
        out = (out - input_zero_point) * input_scale
    else:
        out = tf.cast(out, tf.float32)

    out = np.squeeze(out)
    out = (((out + 1) / 2) * 255)
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    return out


if __name__ == "__main__":
    orig_image, processed_image = load_image(IMAGE_FILENAME)
    start = time.time()
    out = run_inference(interpreter, processed_image)
    end = time.time()
    print("Inference time", end - start, "ms")
    cv2.imwrite('example_output.png', out)
    cv2.imshow('', out)
    cv2.waitKey()