# Copyright 2022-2023 NXP

import tensorflow as tf
import cv2
import numpy as np


def cosine_similarity(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b))

interpreter = tf.lite.Interpreter("facenet512_uint8.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


img1 = cv2.imread("face.jpg", 1)
img1 = np.array(img1, dtype=np.uint8)
img1 = img1[None, ...]

img2 = cv2.imread("face2.jpg", 1)
img2 = np.array(img2, dtype=np.uint8)
img2 = img2[None, ...]


interpreter.set_tensor(input_details[0]['index'], img1)
interpreter.invoke()
out1 = interpreter.get_tensor(output_details[0]['index'])

interpreter.set_tensor(input_details[0]['index'], img2)
interpreter.invoke()
out2 = interpreter.get_tensor(output_details[0]['index'])

cos_sim = cosine_similarity(out1[0, ...], out2[0, ...])

print("Cosine similarity distance:", cos_sim)
print("Same face" if cos_sim < 0.3 else "Different face")
