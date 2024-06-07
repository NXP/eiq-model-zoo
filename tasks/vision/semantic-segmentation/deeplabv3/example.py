#!/usr/bin/env python3
# Copyright 2023-2024 NXP
# SPDX-License-Identifier: MIT

import tensorflow as tf
import cv2
import numpy as np
import time
from random import seed, randint
import matplotlib.pyplot as plt

seed(1337)

# Public domain image
# https://upload.wikimedia.org/wikipedia/commons/thumb/f/f6/Stray_Cat%2C_Nafplio.jpg/800px-Stray_Cat%2C_Nafplio.jpg
IMAGE_FILENAME = "example_input.jpg"
MODEL_FILE = "model.tflite"
N_CLASSES = 21
COLORS = [(0, 0, 0)]
COLORS += [(randint(30, 254),
            randint(30, 254),
            randint(30, 254)) for _ in range(N_CLASSES-1)]


def load_image(filename):
    orig_image = cv2.imread(filename, 1)
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (513, 513))
    image = image[..., ::-1]
    image = np.expand_dims(image, axis=0)
    image = (image - 127.5)/127.5
    return orig_image, image


def run_inference(interpreter, image):

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    image = image.astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]['index']).astype(np.float32)
    return out


interpreter = tf.lite.Interpreter(MODEL_FILE)
interpreter.allocate_tensors()

orig_image, processed_image = load_image(IMAGE_FILENAME)

start = time.time()
out = run_inference(interpreter, processed_image)[0, ...]
end = time.time()
print("Inference time", end - start, "ms")

out = np.argmax(out, axis=-1)

display_image = np.zeros((out.shape[0], out.shape[1], 3))

for i in range(N_CLASSES):
    display_image[out == i] = COLORS[i]

orig_size = orig_image.shape[0:2]
mask_resized = cv2.resize(display_image, (orig_size[1], orig_size[0]))

plt.imshow(np.flip(orig_image, axis=-1))
plt.imshow(mask_resized.astype(np.int8), alpha=0.7)
plt.show()
