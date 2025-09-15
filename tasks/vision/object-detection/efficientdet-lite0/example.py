#!/usr/bin/env python3
# Copyright 2024 NXP
# SPDX-License-Identifier: MIT

import random
import time

import cv2
import numpy as np
import tensorflow as tf

random.seed(42)

OBJECT_DETECTOR_TFLITE = "efficientdet-lite0_quant_int8.tflite"
LABELS_FILE = 'coco_labels_list.txt'

IMAGE_FILENAME = 'example_input.jpg'

INFERENCE_IMG_SIZE = 320
SCORE_THRESHOLD = 0.3

with open(LABELS_FILE, 'r') as f:
    COCO_CLASSES = [line.strip() for line in f.readlines()]

interpreter = tf.lite.Interpreter(OBJECT_DETECTOR_TFLITE)
interpreter.allocate_tensors()


def gen_box_colors():
    colors = []
    for _ in range(len(COCO_CLASSES)):
        r = random.randint(100, 255)
        g = random.randint(100, 255)
        b = random.randint(100, 255)
        colors.append((r, g, b))

    return colors


BOX_COLORS = gen_box_colors()


def load_image(filename):
    orig_image = cv2.imread(filename, 1)
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (INFERENCE_IMG_SIZE, INFERENCE_IMG_SIZE))
    image = np.expand_dims(image, axis=0)

    return orig_image, image


def run_inference(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[0]['index'])  # coordinates
    classes = interpreter.get_tensor(output_details[1]['index'])  # classes
    scores = interpreter.get_tensor(output_details[2]['index'])  # score

    return boxes, scores, classes


if __name__ == "__main__":

    orig_image, processed_image = load_image(IMAGE_FILENAME)

    start = time.time()
    boxes, scores, classes = run_inference(interpreter, processed_image)
    end = time.time()

    shp = orig_image.shape

    boxes *= np.array([shp[0], shp[1], shp[0], shp[1]])

    boxes = boxes.astype(np.int32)
    boxes = np.squeeze(boxes, axis=0)
    classes = np.squeeze(classes, axis=0)
    scores = np.squeeze(scores, axis=0)
    print("Inference time", end - start, "ms")
    print("Detected", boxes.shape[0], "object(s)")
    print("Box coordinates:")

    for i in range(boxes.shape[0]):

        score = scores[i]
        if score < SCORE_THRESHOLD:
            continue
        box = boxes[i]

        print(box, end=" ")

        cls = classes[i]
        class_name = COCO_CLASSES[int(cls)]

        color = BOX_COLORS[int(cls)]
        print("class", class_name, end=" ")
        print("score", score)

        cv2.rectangle(orig_image, (int(box[1]), int(box[0])), (int(box[3]), int(box[2])),
                      color, 3)

        cv2.putText(orig_image, f"{class_name} {score:.2f}",
                    (box[1], box[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imwrite('example_output.jpg', orig_image)
