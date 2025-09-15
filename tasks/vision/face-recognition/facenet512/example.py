#!/usr/bin/env python3
# Copyright 2022-2025 NXP
# SPDX-License-Identifier: MIT
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import cv2
import numpy as np
from sklearn.preprocessing import normalize

def cosine_similarity(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b))

interpreter = tf.lite.Interpreter("facenet512_uint8.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_scale, input_zero_point = input_details[0]["quantization"]  
output_scale, output_zero_point = output_details[0]["quantization"] 
def preprocess_image(image_path,input_scale,input_zero_point):
    image=cv2.imread(image_path)
    image = cv2.resize(image, (160, 160))
    image = image[None, ...]/255.0
    image=(image/input_scale+input_zero_point).astype(np.uint8)
    return image
image1 = preprocess_image('face.jpg',input_scale,input_zero_point)

image2= preprocess_image('face2.jpg',input_scale,input_zero_point)

interpreter.set_tensor(input_details[0]['index'], image1)
interpreter.invoke()
out1 = interpreter.get_tensor(output_details[0]['index'])

interpreter.set_tensor(input_details[0]['index'], image2)
interpreter.invoke()
out2 = interpreter.get_tensor(output_details[0]['index'])

cos_sim = cosine_similarity(out1[0, ...], out2[0, ...])
print("Cosine similarity distance:", cos_sim)
print("Same face" if cos_sim < 0.4 else "Different face")
