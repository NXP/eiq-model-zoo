#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright 2023 NXP

import os
import time

import numpy as np
import tensorflow as tf

LABELS = ["left hand", "right hand", "feet", "tongue"]

DATA_PATH = os.path.join(os.getcwd(), "dataset", "A01E_0.bin")
MODEL_PATH = os.path.join(os.getcwd(), "eegTCNet_quant_int8.tflite")

interpreter = tf.lite.Interpreter(MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
start = time.time()
print("Loaded model")

tensor: np.ndarray = np.fromfile(DATA_PATH, np.int8)

tensor = tensor.reshape([1, 1, 22, 1125])

interpreter.set_tensor(input_details[0]["index"], tensor)
interpreter.invoke()
out = interpreter.get_tensor(output_details[0]["index"])
print(out)
out_quant = output_details[0]["quantization"]

out = out.astype(np.float32)
out = out_quant[0] * (out - out_quant[1])

print("Output tensor:", out)
print("Recognized movement:", LABELS[out.argmax()])

end = time.time()
print(f"Time %fms\n" % ((end - start) * 1000))
