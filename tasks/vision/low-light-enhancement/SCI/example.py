#!/usr/bin/env python3
# Copyright 2024 NXP
# SPDX-License-Identifier: MIT

import cv2
import tensorflow as tf
import numpy as np
import time

OBJECT_DETECTOR_TFLITE = 'sci.tflite'
IMAGE_FILENAME = 'example_input.png'
SIZE_MODEL = (1920, 1080)

interpreter = tf.lite.Interpreter(OBJECT_DETECTOR_TFLITE)
interpreter.allocate_tensors()


def load_image(filename, size):
    orig_image = cv2.imread(filename, 1)
    image = cv2.resize(orig_image, (size[0], size[1]))
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return orig_image, image


def run_inference(interpreter, image):

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    type_tf = input_details[0]['dtype']
    if type_tf == np.int8:
        input_scale, input_zero_point = input_details[0]["quantization"]
        image = image / input_scale + input_zero_point
        image = image.astype(np.int8)
    else:
        image = image.astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()

    i = interpreter.get_tensor(output_details[0]['index'])
    r = interpreter.get_tensor(output_details[1]['index'])
    return i, r


def postprocessing(out, inp, orig_image):
    out_final = inp / tf.squeeze(out)
    out_final = tf.clip_by_value(out_final, 0, 1)
    out_final = tf.squeeze(out_final).numpy()
    shp = orig_image.shape
    out_final = cv2.resize(out_final, (shp[1], shp[0])) * 255.0
    return out_final


if __name__ == "__main__":
    orig_image, processed_image = load_image(IMAGE_FILENAME, SIZE_MODEL)
    start = time.time()
    out, _ = run_inference(interpreter, processed_image)
    end = time.time()
    out_ps = postprocessing(out, processed_image, orig_image)
    cv2.imwrite('example_output.png', out_ps)
    cv2.imshow('', orig_image)
    cv2.waitKey()
