#!/usr/bin/env python3
# Copyright 2022-2024 NXP
# SPDX-License-Identifier: MIT

import cv2
import tensorflow as tf
import numpy as np
import time
import math
import random

random.seed(1337)

OBJECT_DETECTOR_TFLITE = "nanodet_m_0.5x.tflite"
IMAGE_FILENAME = "example_input.jpg"
SCORE_THRESHOLD = 0.20
IOU_THRESHOLD = 0.5
REG_MAX = 7
MODEL_STRIDE = 32
INFERENCE_IMG_SIZE = 320
MAX_DETS = 100
COCO_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard',
                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
                'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
                'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
                'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                'scissors', 'teddy bear', 'hair drier', 'toothbrush']

interpreter = tf.lite.Interpreter(OBJECT_DETECTOR_TFLITE)
interpreter.allocate_tensors()


def gen_box_colors():
    colors = []
    for _ in range(80):
        r = random.randint(100, 255)
        g = random.randint(100, 255)
        b = random.randint(100, 255)
        colors.append((r, g, b))

    return colors


BOX_COLORS = gen_box_colors()


def normalize(img):
    mean = [103.53, 116.28, 123.675]
    std = [57.375, 57.12, 58.395]
    mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3) / 255
    std = np.array(std, dtype=np.float32).reshape(1, 1, 3) / 255
    img = (img - mean) / std
    return img


def load_image(filename):
    orig_image = cv2.imread(filename, 1)
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (INFERENCE_IMG_SIZE, INFERENCE_IMG_SIZE))
    image = image[..., ::-1]
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    image = normalize(image)
    return orig_image, image


def generate_center_priors(input_height, input_width, stride):

    feat_w = math.ceil(float(input_width) / stride)
    feat_h = math.ceil(float(input_height) / stride)

    centers = [[x+0.5, y+0.5] for y in range(feat_h) for x in range(feat_w)]
    return np.array(centers) * stride


center_priors = generate_center_priors(INFERENCE_IMG_SIZE,
                                       INFERENCE_IMG_SIZE,
                                       MODEL_STRIDE)


def softmax_np(x):
    '''
    Compute softmax on the last dim of numpy array
    '''
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def decode_boxes_prediction(boxes):
    '''
    Decoding boxes as in Generalized Focal Loss paper
    https://arxiv.org/pdf/2006.04388.pdf
    '''
    boxes = softmax_np(boxes)
    boxes = boxes * np.linspace(0, REG_MAX, num=REG_MAX+1)
    boxes = np.sum(boxes, axis=-1)

    boxes *= MODEL_STRIDE

    boxes[..., 0] = np.maximum(center_priors[:, 0] - boxes[..., 0], 0)
    boxes[..., 1] = np.maximum(center_priors[:, 1] - boxes[..., 1], 0)
    boxes[..., 2] = np.minimum(center_priors[:, 0] + boxes[..., 2],
                               INFERENCE_IMG_SIZE)
    boxes[..., 3] = np.minimum(center_priors[:, 1] + boxes[..., 3],
                               INFERENCE_IMG_SIZE)

    return boxes


def decode_output(scores, boxes,
                  score_threshold=SCORE_THRESHOLD,
                  iou_threshold=IOU_THRESHOLD):
    '''
    Decode output from nanodet in inference size referential (e.g 320x320)
    '''

    # 80 class scores per prior box
    scores = scores.reshape((-1, len(COCO_CLASSES)))

    # 8 scores per coordinate per prior box
    boxes = boxes.reshape((-1, 4, REG_MAX+1))

    boxes = decode_boxes_prediction(boxes)

    # find maximum class for each box
    classes = np.argmax(scores, axis=-1)

    # find maximum score for each box
    scores = np.max(scores, axis=-1)

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
    input_scale, input_zero_point = input_details[0]["quantization"]

    image = image / input_scale + input_zero_point
    image = image.astype(np.int8)

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()

    scores_scale, scores_zero_point = output_details[0]["quantization"]
    boxes_scale, boxes_zero_point = output_details[1]["quantization"]

    scores = interpreter.get_tensor(output_details[0]['index']).astype(np.int8)
    boxes = interpreter.get_tensor(output_details[1]['index']).astype(np.int8)

    # dequantize output
    scores = (scores - scores_zero_point) * scores_scale
    boxes = (boxes - boxes_zero_point) * boxes_scale

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
        cv2.putText(orig_image,  f"{class_name} {score}",
                    (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow('', orig_image)
    cv2.waitKey()
