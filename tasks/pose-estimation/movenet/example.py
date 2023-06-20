#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright 2022-2023 NXP

import cv2
import tensorflow as tf
import numpy as np
import time

MODEL_FILENAME = "movenet.tflite"
IMAGE_FILENAME = "example_input.jpg"
LINE_COLOR = (255, 128, 0)
POINT_COLOR = (0, 0, 255)

# keypoints definition
# https://github.com/tensorflow/tfjs-models/tree/master/pose-detection#keypoint-diagram
keypoints_def = [
    {'label': 'nose',               'connections': [1, 2,]},
    {'label': 'left_eye',           'connections': [0, 3,]},
    {'label': 'right_eye',          'connections': [0, 4,]},
    {'label': 'left_ear',           'connections': [1,]},
    {'label': 'right_ear',          'connections': [2,]},
    {'label': 'left_shoulder',      'connections': [6, 7, 11,]},
    {'label': 'right_shoulder',     'connections': [5, 8, 12,]},
    {'label': 'left_elbow',         'connections': [5, 9,]},
    {'label': 'right_elbow',        'connections': [6, 10,]},
    {'label': 'left_wrist',         'connections': [7,]},
    {'label': 'right_wrist',        'connections': [8,]},
    {'label': 'left_hip',           'connections': [5, 12, 13,]},
    {'label': 'right_hip',          'connections': [6, 11, 14,]},
    {'label': 'left_knee',          'connections': [11, 15,]},
    {'label': 'right_knee',         'connections': [12, 16,]},
    {'label': 'left_ankle',         'connections': [13,]},
    {'label': 'right_ankle',        'connections': [14,]},
    ]

connections = [(i, j) for i in range(len(keypoints_def))
               for j in keypoints_def[i]["connections"]]

interpreter = tf.lite.Interpreter(MODEL_FILENAME)
interpreter.allocate_tensors()


def load_image(filename):
    orig_image = cv2.imread(filename, 1)
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (192, 192))
    image = image[..., ::-1]
    image = np.expand_dims(image, axis=0)
    return orig_image, image


def run_inference(interpreter, image):

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    image = image.astype(np.uint8)

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]['index']).astype(np.float32)

    return out


orig_image, processed_image = load_image(IMAGE_FILENAME)

start = time.time()

out = run_inference(interpreter, processed_image)[0, 0, ...]

end = time.time()
print("Inference time", end - start, "ms")

w, h = orig_image.shape[0:2]
out[:, 0] *= w
out[:, 1] *= h

for c in connections:
    i = c[0]
    j = c[1]
    cv2.line(orig_image,
             (int(out[i, 1]), int(out[i, 0])),
             (int(out[j, 1]), int(out[j, 0])), LINE_COLOR, 5)

for i in range(out.shape[0]):
    cv2.circle(orig_image,
               (int(out[i, 1]), int(out[i, 0])), 5, POINT_COLOR, 10)

cv2.imwrite("example_output.jpg", orig_image)
cv2.imshow('', orig_image)
cv2.waitKey()
