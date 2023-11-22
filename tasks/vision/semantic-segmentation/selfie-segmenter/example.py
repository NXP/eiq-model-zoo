#!/usr/bin/env python3

"""
Copyright 2023 NXP
SPDX-License-Identifier: MIT

Model: MediaPipe's Selfie Segmenter
Model licensed under Apache-2.0 License

Original model available at:
    https://developers.google.com/mediapipe/solutions/vision/image_segmenter

Model Card:
    https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Selfie%20Segmentation.pdf

Model created by:
    Tingbo Hou, Google; Siargey Pisarchyk, Google; Karthik Raveendran, Google.

Example script to show human segmentation from a given picture
"""

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite


def normalize_input(input_data, input_shape):
    """Fit the image size for model and change colorspace"""
    input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
    resized_data = cv2.resize(input_data, input_shape)
    normalized_data = np.ascontiguousarray(resized_data / 255.0)
    normalized_data = normalized_data.astype("float32")
    normalized_data = normalized_data[None, ...]
    return normalized_data


# Load model
model_path = "./selfie_segmenter_int8.tflite"
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]["index"]
input_shape = interpreter.get_input_details()[0]["shape"]
output = interpreter.get_output_details()[0]["index"]

# Load input image and normalize it
frame = cv2.imread("./data/example_input.jpg")
input_frame = normalize_input(frame, (input_shape[2], input_shape[1]))

# Perform inference
interpreter.set_tensor(input_index, input_frame)
interpreter.invoke()
mask = interpreter.get_tensor(output)[0]

# Resize output mask and set condition of detection
mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), cv2.INTER_CUBIC)
condition = np.stack((mask,) * 3, axis=-1) > 0.1

# Generate foreground and background for segmentation mask.
foreground = np.full(shape=frame.shape, fill_value=255, dtype=np.uint8)
background = np.full(shape=frame.shape, fill_value=0, dtype=np.uint8)

segmentation = np.where(condition, foreground, background)
cv2.imwrite("./data/example_output.jpg", segmentation)
