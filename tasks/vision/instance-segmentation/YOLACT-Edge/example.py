#!/usr/bin/env python3
# Copyright 2024 NXP
# SPDX-License-Identifier: MIT

import cv2
import tensorflow as tf
import numpy as np
import time
import math
import random

random.seed(1337)

OBJECT_DETECTOR_TFLITE = "YOLACT-Edge.tflite"
IMAGE_FILENAME = "example_input.jpg"
SCORE_THRESHOLD = 0.2
IOU_THRESHOLD = 0.5
INFERENCE_IMG_SIZE = 550
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


def load_image(filename):
    orig_image = cv2.imread(filename, 1)
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (INFERENCE_IMG_SIZE, INFERENCE_IMG_SIZE))
    image = image[..., ::-1]
    image = np.expand_dims(image, axis=0)
    return orig_image, image


def decode_output(scores, boxes, masks, proto,
                  score_threshold=SCORE_THRESHOLD,
                  iou_threshold=IOU_THRESHOLD):
    '''
    Decode output from YOLACT-Edge in inference size referential (550x550)
    '''
    # squeeze each output of the model
    scores = scores[0].T
    boxes = boxes[0]
    masks = masks[0]
    proto = proto[0]

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
    masks = tf.gather(masks, inds)

    # computation of the final masks, the equation is available in the paper
    final_masks = tf.linalg.matmul(proto, tf.transpose(masks))
    final_masks = tf.transpose(tf.sigmoid(final_masks), perm=[2, 0, 1])
    return scores, boxes, classes, final_masks


def run_inference(interpreter, image, threshold=SCORE_THRESHOLD):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # scale the image according the quantization formula
    if input_details[0]['dtype'] == tf.int8:
        input_scale, input_zero_point = input_details[0]["quantization"]
        image = image / input_scale + input_zero_point
        image = image.astype(np.int8)
    else:
        image = image.astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()

    # get the four components of the output of the model
    boxes = interpreter.get_tensor(output_details[0]['index'])
    scores = interpreter.get_tensor(output_details[1]['index'])
    proto = interpreter.get_tensor(output_details[2]['index'])
    masks = interpreter.get_tensor(output_details[3]['index'])
    return boxes, proto, scores, masks


if __name__ == "__main__":

    orig_image, processed_image = load_image(IMAGE_FILENAME)

    start = time.time()
    boxes, proto, scores, masks = run_inference(interpreter, processed_image)
    end = time.time()

    scores, boxes, classes, masks = decode_output(scores, boxes, masks, proto)

    # rescale boxes for display
    shp = orig_image.shape
    boxes = boxes.numpy()
    bboxes_final = boxes * np.array([shp[1], shp[0], shp[1], shp[0]])
    bboxes_final = bboxes_final.astype(np.int32)

    masks_list = []

    print("Inference time", end - start, "ms")
    print("Detected", boxes.shape[0], "object(s)")
    print("Box coordinates:")
    for i in range(boxes.shape[0]):
        box = boxes[i, :]
        score = scores[i].numpy()
        color = BOX_COLORS[classes[i].numpy()]
        id_class = int(classes[i])
        mask_image = masks[i]

        # steps to draw the final masks. We have to crop them inside the
        # bounding boxes
        shp_mask = mask_image.shape
        box_mask = box * \
            np.array([shp_mask[1], shp_mask[0], shp_mask[1], shp_mask[0]])
        box_mask = box_mask.astype(np.int32)
        region = np.zeros(shp_mask, dtype=np.float32)
        x1, y1, x2, y2 = box_mask
        width, height = x2 - x1, y2 - y1
        region[y1:y2, x1:x2] = (mask_image > 0.5)[y1:y2, x1:x2]
        region = np.expand_dims(region, axis=-1) * np.ones((1, 1, 3))
        region = np.where(region[:, :] == 0, 0, color).astype(np.float32)
        resized_mask = cv2.resize(
            region, (shp[1], shp[0]), cv2.INTER_NEAREST).astype(
            np.uint8)
        box = bboxes_final[i, :]
        print(box, end=" ")
        class_name = COCO_CLASSES[classes[i].numpy()]
        print("class", class_name, end=" ")
        print("score", score)

        # the masks are added to the original image with adequate transparency
        # values
        cv2.addWeighted(orig_image, 1, resized_mask, 0.5, 0, orig_image)

        cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]),
                      color, 3)
        cv2.putText(orig_image, f"{class_name} {score:.2f}",
                    (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.imwrite('example_output.jpg', orig_image)
    cv2.imshow('', orig_image)
    cv2.waitKey()