#!/usr/bin/env python3
# Copyright 2023-2024 NXP
# SPDX-License-Identifier: MIT

import cv2
import argparse
import numpy as np
import time
import tensorflow as tf
from labels import IMAGENET_LABELS


parser = argparse.ArgumentParser()
parser.add_argument("--model_path",
                    default="efficientnet-lite0-int8.tflite",
                    type=str)
parser.add_argument("--image", default="example_input.jpg", type=str)
args = parser.parse_args()


interpreter = tf.lite.Interpreter(args.model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Loaded model")

im = cv2.imread(args.image, 1)
start = time.time()

im = cv2.resize(im, (224, 224))

im = im[None, ...] / 127.5 - 1.0

input_scale, input_zero_point = input_details[0]["quantization"]
im = im / input_scale + input_zero_point

im = im.astype(np.uint8)

interpreter.set_tensor(input_details[0]['index'], im)
interpreter.invoke()
out = interpreter.get_tensor(output_details[0]['index'])
print("Output tensor:", out)
print("Recognized object:", IMAGENET_LABELS[out.argmax()-1])

end = time.time()
print(f"Time %fms\n" % ((end - start)*1000))
