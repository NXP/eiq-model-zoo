#!/usr/bin/env python3
# Copyright 2022-2024 NXP
# SPDX-License-Identifier: MIT

import cv2
import tensorflow as tf
import numpy as np
import time

FACE_DETECTOR_TFLITE = "ultraface_slim_uint8_float32.tflite"
IMAGE_FILENAME = "example_input.jpg"
BOX_COLOR = (255, 128, 0)
THRESHOLD = 0.5

interpreter = tf.lite.Interpreter(FACE_DETECTOR_TFLITE)
interpreter.allocate_tensors()


def load_image(filename):
    orig_image = cv2.imread(filename, 1)
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (320, 240))
    image = image[..., ::-1]
    image = np.expand_dims(image, axis=0)
    image = (image - 127.0)/128.0
    return orig_image, image


def run_inference(interpreter, image, threshold=THRESHOLD):

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_scale, input_zero_point = input_details[0]["quantization"]

    image = image / input_scale + input_zero_point
    image = image.astype(np.uint8)

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]['index']).astype(np.float32)

    boxes = out[..., 2:]
    scores = out[..., 1]

    conf_mask = threshold < scores
    boxes, scores = boxes[conf_mask], scores[conf_mask]

    return boxes, scores


orig_image, processed_image = load_image(IMAGE_FILENAME)

start = time.time()
boxes, scores = run_inference(interpreter, processed_image)
boxes *= np.tile(orig_image.shape[1::-1], 2)
boxes = boxes.astype(np.int32)

end = time.time()
print("Inference time", end - start, "ms")
print("Detected", boxes.shape[0], "faces")
print("Box coordinates:")
for i in range(boxes.shape[0]):
    box = boxes[i, :]
    print(box)
    cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), BOX_COLOR, 4)
cv2.imshow('', orig_image)
cv2.waitKey()
