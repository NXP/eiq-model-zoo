#!/usr/bin/env python3
# Copyright 2023 NXP
# SPDX-License-Identifier: MIT

import cv2
import tensorflow as tf
import numpy as np
import time
import random

random.seed(42)

OBJECT_DETECTOR_TFLITE = 'ssdlite-mobilenetv2.tflite'
LABELS_FILE = 'coco_labels_list.txt'
BOX_PRIORS_FILE = 'box_priors.txt'
IMAGE_FILENAME = 'example_input.jpg'

SCORE_THRESHOLD = 0.20
NMS_IOU_THRESHOLD = 0.5
INFERENCE_IMG_SIZE = 300
MAX_DETS = 100

# scales for bounding box decoding
Y_SCALE = 10.0
X_SCALE = 10.0
H_SCALE = 5.0
W_SCALE = 5.0

with open(LABELS_FILE, 'r') as f:
    COCO_CLASSES = [line.strip() for line in f.readlines()]

BOX_PRIORS = np.loadtxt(BOX_PRIORS_FILE).T

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
    image = image / 127.5 - 1
    return orig_image, image


def decode_boxes_prediction(boxes):

    # decode according to equation 2 from https://arxiv.org/pdf/1512.02325.pdf
    y_center = boxes[..., 0] / Y_SCALE * BOX_PRIORS[:, 2] + BOX_PRIORS[:, 0]
    x_center = boxes[..., 1] / X_SCALE * BOX_PRIORS[:, 3] + BOX_PRIORS[:, 1]
    h = np.exp(boxes[..., 2] / H_SCALE) * BOX_PRIORS[:, 2]
    w = np.exp(boxes[..., 3] / W_SCALE) * BOX_PRIORS[:, 3]

    boxes[..., 0] = x_center - w / 2.0
    boxes[..., 1] = y_center - h / 2.0
    boxes[..., 2] = x_center + w / 2.0
    boxes[..., 3] = y_center + h / 2.0

    boxes *= INFERENCE_IMG_SIZE

    return boxes


def decode_output(scores, boxes,
                  score_threshold=SCORE_THRESHOLD,
                  iou_threshold=NMS_IOU_THRESHOLD):
    '''
    Decode output from SSD MobileNet V2 in inference size referential (300x300)
    '''

    # 91 class scores per prior box
    scores = scores.reshape((-1, len(COCO_CLASSES)))

    # 4 coordinates per prior box
    boxes = boxes.reshape((-1, 4))

    boxes = decode_boxes_prediction(boxes)

    # find maximum class score for each box
    # while ignoring first class
    classes = np.argmax(scores[:, 1:], axis=-1) + 1

    # find maximum score for each box
    scores = np.max(scores, axis=-1)
    # apply sigmoid to convert logits to probabilities
    scores = 1 / (1 + np.exp(scores))

    # apply NMS from tensorflow
    inds = tf.image.non_max_suppression(boxes, scores, MAX_DETS,
                                        score_threshold=score_threshold,
                                        iou_threshold=iou_threshold)

    # keep only selected boxes
    boxes = tf.gather(boxes, inds)
    scores = tf.gather(scores, inds)
    classes = tf.gather(classes, inds)

    return scores, boxes, classes


def run_inference(interpreter, image, threshold=SCORE_THRESHOLD):

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    image = image.astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[0]['index'])
    scores = interpreter.get_tensor(output_details[1]['index'])

    return boxes, scores


if __name__ == "__main__":

    orig_image, processed_image = load_image(IMAGE_FILENAME)

    start = time.time()
    boxes, scores = run_inference(interpreter, processed_image)
    end = time.time()

    scores, boxes, classes = decode_output(scores, boxes)

    # rescale boxes for display
    shp = orig_image.shape
    boxes = boxes.numpy()
    boxes /= INFERENCE_IMG_SIZE
    boxes *= np.array([shp[1], shp[0], shp[1], shp[0]])

    boxes = boxes.astype(np.int32)

    print("Inference time", end - start, "ms")
    print("Detected", boxes.shape[0], "object(s)")
    print("Box coordinates:")

    for i in range(boxes.shape[0]):
        box = boxes[i, :]
        print(box, end=" ")
        class_name = COCO_CLASSES[classes[i].numpy()]
        score = scores[i].numpy()
        color = BOX_COLORS[classes[i]]
        print("class", class_name, end=" ")
        print("score", score)
        cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]),
                      color, 3)
        cv2.putText(orig_image,  f"{class_name} {score:.2f}",
                    (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imwrite('example_output.jpg', orig_image)
    cv2.imshow('', orig_image)
    cv2.waitKey()
