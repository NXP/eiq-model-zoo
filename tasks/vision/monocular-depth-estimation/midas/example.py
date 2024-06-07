#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright 2023-2024 NXP

from argparse import ArgumentParser
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time


WIDTH = 256
HEIGHT = 256

parser = ArgumentParser()

parser.add_argument(
    "-m", "--model",
    help="Path to a .tflite file.",
    type=str,
    default="midas_2_1_small_int8.tflite")

parser.add_argument(
    "-i", "--input",
    help="Path to a input image file",
    type=str,
    default="example_input.jpg")

args = parser.parse_args()


def load_image(filename):
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    orig_height, orig_width = image.shape[:-1]
    image = cv2.resize(image, (WIDTH, HEIGHT), cv2.INTER_CUBIC)
    image = image.astype(np.float32)
    image = image / 255
    image_input = np.expand_dims(image, 0)
    return image_input, orig_height, orig_width


def run_inference(interpreter, image):

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    image = image.astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]['index']).astype(np.float32)
    return out


def post_process(outputs, orig_height, orig_width):
    disp = cv2.resize(outputs, (orig_width, orig_height), cv2.INTER_CUBIC)

    # rescale disp
    disp_min = disp.min()
    disp_max = disp.max()

    if disp_max - disp_min > 1e-6:
        disp = (disp - disp_min) / (disp_max - disp_min)
    else:
        disp.fill(0.5)

    return disp


interpreter = tf.lite.Interpreter(args.model)
interpreter.allocate_tensors()

image_input, orig_height, orig_width = load_image(args.input)

start = time.time()
out = run_inference(interpreter, image_input)[0, ...]
end = time.time()
print("Inference time", end - start, "ms")

disp = post_process(out, orig_height, orig_width)

# pfm
out = 'example_output.pfm'
cv2.imwrite(out, disp)

# png
out = 'example_output.jpg'
plt.imsave(out, disp, vmin=0, vmax=1, cmap='inferno')
