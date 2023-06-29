#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright 2023 NXP

import cv2
import tensorflow as tf
import numpy as np
import time
import random

random.seed(42)

OBJECT_DETECTOR_TFLITE = 'yolov4-tiny_416_quant.tflite'
LABELS_FILE = 'coco-labels-2014_2017.txt'
IMAGE_FILENAME = 'example_input.jpg'

SCORE_THRESHOLD = 0.20
NMS_IOU_THRESHOLD = 0.5
INFERENCE_IMG_SIZE = 416
MAX_DETS = 100

ANCHORS = [[[81, 82], [135, 169], [344, 319]], [[23, 27], [37, 58], [81, 82]]]
SIGMOID_FACTOR = [1.05, 1.05]
NUM_ANCHORS = 3
STRIDES = [32, 16]
GRID_SIZES = [int(INFERENCE_IMG_SIZE / s) for s in STRIDES]

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
    image = image / 255.0
    return orig_image, image


def np_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def reciprocal_sigmoid(x):
    return -np.log(1 / x - 1)


def decode_boxes_prediction(yolo_output):
    # Each output level represents a grid of predictions.
    # The first output level is a 26x26 grid and the second 13x13.
    # Each cell of each grid is assigned to 3 anchor bounding boxes.
    # The bounding box predictions are regressed
    # relatively to these anchor boxes.
    # Thus, the model predicts 3 bounding boxes per cell per output level.
    # The output is structured as follows:
    # For each cell [[x, y, w, h, conf, cl_0, cl_1, ..., cl_79], # anchor 1
    #                [x, y, w, h, conf, cl_0, cl_1, ..., cl_79], # anchor 2
    #                [x, y, w, h, conf, cl_0, cl_1, ..., cl_79]] # anchor 3
    # Hence, we have 85 values per anchor box, and thus 255 values per cell.
    # The decoding of the output bounding boxes is described in Figure 2 of
    # the YOLOv3 paper https://arxiv.org/pdf/1804.02767.pdf;

    boxes_list = []
    scores_list = []
    classes_list = []

    for idx, feats in enumerate(yolo_output):

        features = np.reshape(feats, (NUM_ANCHORS * GRID_SIZES[idx] ** 2, 85))

        anchor = np.array(ANCHORS[idx])
        factor = SIGMOID_FACTOR[idx]
        grid_size = GRID_SIZES[idx]
        stride = STRIDES[idx]

        cell_confidence = features[..., 4]
        logit_threshold = reciprocal_sigmoid(SCORE_THRESHOLD)
        over_threshold_list = np.where(cell_confidence > logit_threshold)

        if over_threshold_list[0].size > 0:
            indices = np.array(over_threshold_list[0])

            box_positions = np.floor_divide(indices, 3)

            list_xy = np.array(np.divmod(box_positions, grid_size)).T
            list_xy = list_xy[..., ::-1]
            boxes_xy = np.reshape(list_xy, (int(list_xy.size / 2), 2))

            outxy = features[indices, :2]

            # boxes center coordinates
            centers = np_sigmoid(outxy * factor) - 0.5 * (factor - 1)
            centers += boxes_xy
            centers *= stride

            # boxes width and height
            width_height = np.exp(features[indices, 2:4])
            width_height *= anchor[np.divmod(indices, NUM_ANCHORS)[1]]

            boxes_list.append(np.stack([centers[:, 0] - width_height[:, 0]/2,
                                        centers[:, 1] - width_height[:, 1]/2,
                                        centers[:, 0] + width_height[:, 0]/2,
                                        centers[:, 1] + width_height[:, 1]/2],
                                       axis=1))

            # confidence that cell contains an object
            scores_list.append(np_sigmoid(features[indices, 4:5]))

            # class with the highest probability in this cell
            classes_list.append(np.argmax(features[indices, 5:], axis=1))

    if len(boxes_list) > 0:
        boxes = np.concatenate(boxes_list, axis=0)
        scores = np.concatenate(scores_list, axis=0)[:, 0]
        classes = np.concatenate(classes_list, axis=0)

        return boxes, scores, classes
    else:
        return np.zeros((0, 4)), np.zeros((0)), np.zeros((0))


def decode_output(yolo_outputs,
                  score_threshold=SCORE_THRESHOLD,
                  iou_threshold=NMS_IOU_THRESHOLD):
    '''
    Decode output from YOLOv4 tiny in inference size referential (416x416)
    '''
    boxes, scores, classes = decode_boxes_prediction(yolo_outputs)

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

    boxes = interpreter.get_tensor(output_details[0]['index'])
    boxes2 = interpreter.get_tensor(output_details[1]['index'])

    return [boxes, boxes2]


if __name__ == "__main__":

    orig_image, processed_image = load_image(IMAGE_FILENAME)

    start = time.time()
    yolo_output = run_inference(interpreter, processed_image)
    end = time.time()

    scores, boxes, classes = decode_output(yolo_output)

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
