#!/usr/bin/env python3
# Copyright 2023 NXP
# SPDX-License-Identifier: MIT

import argparse
import time

import cv2
import numpy as np
import tensorflow as tf

CLASSES = ['non-person', 'person']

parser = argparse.ArgumentParser()
parser.add_argument("--model_path",
                    default="vww_96_int8.tflite",
                    type=str)
parser.add_argument("--image", default="example_input.jpg", type=str)
args = parser.parse_args()

interpreter = tf.lite.Interpreter(args.model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Loaded model")

img = cv2.imread(args.image, 1)
start = time.time()

img = cv2.resize(img, (96, 96))

img = np.array(img, dtype=np.int64)
img = img - 128
img = img.astype(np.int8)
img = np.expand_dims(img, axis=0)

interpreter.set_tensor(input_details[0]['index'], img)
interpreter.invoke()
out = interpreter.get_tensor(output_details[0]['index'])

out_quant = output_details[0]["quantization"]
detection_classes = out.astype(np.float32)
detection_classes = out_quant[0] * (detection_classes - out_quant[1])
print("Output tensor:", detection_classes)
print("Recognized object:", CLASSES[detection_classes.argmax()])

end = time.time()
print(f"Time %fms\n" % ((end - start) * 1000))
