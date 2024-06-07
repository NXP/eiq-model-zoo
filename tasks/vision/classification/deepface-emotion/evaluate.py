#!/usr/bin/env python3
# Copyright 2022-2024 NXP
# SPDX-License-Identifier: MIT

import argparse
import numpy as np
import tensorflow as tf
from deepface.extendedmodels.Emotion import loadModel
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

parser = argparse.ArgumentParser()
parser.add_argument("--model_path",
                    default="emotion_uint8_float32.tflite", type=str)
parser.add_argument("--test_path", default="fer2013/fer2013.csv", type=str)
args = parser.parse_args()


interpreter = tf.lite.Interpreter(args.model_path)
interpreter.allocate_tensors()
model = loadModel()
print("Loaded models")


def run_inference(interpreter, keras_model, im):

    out_keras = keras_model(im)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_scale, input_zero_point = input_details[0]["quantization"]
    im_scaled = im / input_scale + input_zero_point

    im_scaled = im_scaled.astype(np.uint8)

    interpreter.set_tensor(input_details[0]['index'], im_scaled)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]['index'])
    return out, out_keras


split = "PrivateTest"

with open(args.test_path, 'r') as f:
    lines = f.readlines()

lines = [li.strip().split(",") for li in lines[1:]]
lines = filter(lambda l: l[2] == "PrivateTest")
lines = [(int(li[0]), np.array(li[1].split(" "))) for li in lines]
lines = [(li[0], li[1].reshape((48, 48, 1))) for li in lines]
lines = [(li[0], li[1].astype(np.float32)) for li in lines]

n_correct_tflite = 0
n_correct_keras = 0

true_labels = []
predictions = []

for l in tqdm(lines):
    out, out_keras = run_inference(interpreter, model, l[1][None, ...] / 255.0)

    true_labels.append(l[0])
    predictions.append(out.argmax())

    if out.argmax() == l[0]:
        n_correct_tflite += 1

    if out_keras.numpy().argmax() == l[0]:
        n_correct_keras += 1


print("TensorFlow Lite model accuracy", n_correct_tflite / len(lines))
print("Keras model accuracy", n_correct_keras / len(lines))

cm = confusion_matrix(true_labels, predictions, normalize='pred')

disp = ConfusionMatrixDisplay(cm, display_labels=LABELS)
disp.plot()
plt.show()
