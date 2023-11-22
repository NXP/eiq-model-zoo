#!/usr/bin/env python3
# Copyright 2023 NXP
# SPDX-License-Identifier: MIT

import cv2
import argparse
import numpy as np
import time
import tensorflow as tf
from utils import draw_axis, normalize, decode

parser = argparse.ArgumentParser()
parser.add_argument("--model_path",
                    default="whenet.tflite",
                    type=str)
parser.add_argument("--image", default="example_input.jpg", type=str)
args = parser.parse_args()


interpreter = tf.lite.Interpreter(args.model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Loaded model")

im = cv2.imread(args.image, 1)

orig_image = im

im = cv2.resize(im, (224, 224))

im = normalize(im)

input_scale, input_zero_point = input_details[0]["quantization"]
im = im / input_scale + input_zero_point

im = im.astype(np.int8)

interpreter.set_tensor(input_details[0]['index'], im)

start = time.time()
interpreter.invoke()
end = time.time()
print(f"Time %fms\n" % ((end - start)*1000))

r = interpreter.get_tensor(output_details[2]['index'])
y = interpreter.get_tensor(output_details[1]['index'])
p = interpreter.get_tensor(output_details[0]['index'])

output_scale_roll, output_zero_point_roll = output_details[2]["quantization"]
output_scale_yaw, output_zero_point_yaw = output_details[1]["quantization"]
output_scale_pitch, output_zero_point_pitch = output_details[0]["quantization"]

r = (r - output_zero_point_roll) * output_scale_roll
y = (y - output_zero_point_yaw) * output_scale_yaw
p = (p - output_zero_point_pitch) * output_scale_pitch

yaw_predicted, pitch_predicted, roll_predicted = decode(y, p, r)

print("Yaw", yaw_predicted,
      "Pitch", pitch_predicted,
      "Roll", roll_predicted)

draw_axis(orig_image, yaw_predicted, pitch_predicted, roll_predicted)
cv2.imshow('output', orig_image)
cv2.waitKey(5000)
cv2.imwrite('example_output.jpg', orig_image)
