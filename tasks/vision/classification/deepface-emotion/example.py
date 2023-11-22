#!/usr/bin/env python3
# Copyright 2022-2023 NXP
# SPDX-License-Identifier: MIT

import cv2
import argparse
import numpy as np
import time
import tensorflow as tf

LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

parser = argparse.ArgumentParser()
parser.add_argument("--model_path",
                    default="emotion_uint8_float32.tflite",
                    type=str)
parser.add_argument("--image", default="test.jpg", type=str)
args = parser.parse_args()


interpreter = tf.lite.Interpreter(args.model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Loaded model")

im = cv2.imread(args.image, 0)
start = time.time()

im = cv2.resize(im, (48, 48))

im = im[None, ..., None] / 255.0

input_scale, input_zero_point = input_details[0]["quantization"]
im = im / input_scale + input_zero_point

im = im.astype(np.uint8)

interpreter.set_tensor(input_details[0]['index'], im)
interpreter.invoke()
out = interpreter.get_tensor(output_details[0]['index'])
print("Output tensor:", out)
print("Recognized emotion:", LABELS[out.argmax()])

end = time.time()
print(f"Time %fms\n" % ((end - start)*1000))
