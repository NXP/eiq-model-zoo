#!/usr/bin/env python3
# Copyright 2024 NXP
# SPDX-License-Identifier: MIT

import wget
import zipfile
import json
import glob
import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from example import load_image, decode_output
from example import run_inference, gen_box_colors, decode_output_nms

OBJECT_DETECTOR_TFLITE = "centernet.tflite"
SCORE_THRESHOLD = 0.00
NMS_IOU_THRESHOLD = 0.5
COCO_WEBSITE = "http://images.cocodataset.org"
VAL_ANNO_URL = COCO_WEBSITE + "/annotations/annotations_trainval2017.zip"
VAL_IMG_URL = COCO_WEBSITE + "/zips/val2017.zip"

BOX_COLORS = gen_box_colors()

if not os.path.exists("coco/val2017"):
    response = wget.download(VAL_IMG_URL, "val2017.zip")
    with zipfile.ZipFile("val2017.zip", 'r') as zip_ref:
        zip_ref.extractall("coco")

if not os.path.exists("coco/annotations"):
    response = wget.download(VAL_ANNO_URL, "annotations_trainval2017.zip")
    with zipfile.ZipFile("annotations_trainval2017.zip", 'r') as zip_ref:
        zip_ref.extractall("coco")


interpreter = tf.lite.Interpreter(OBJECT_DETECTOR_TFLITE)
interpreter.allocate_tensors()

annotations = COCO(annotation_file="coco/annotations/instances_val2017.json")


def evaluate(interpreter):
    image_filenames = glob.glob("coco/val2017/*")

    results = []

    for image_fn in tqdm(image_filenames, desc="Evaluating"):

        image_id = int(os.path.splitext(os.path.basename(image_fn))[0])
        orig_image, processed_image = load_image(image_fn)

        heatmap, size, offset, processed_image = run_inference(
            interpreter, processed_image)

        boxes, scores, classes = decode_output(heatmap, size, offset)
        boxes = tf.squeeze(boxes)
        scores = tf.squeeze(scores)
        classes = tf.squeeze(classes)

        # apply tensorflow's implementation of nms to the outputs.
        boxes, scores, classes = decode_output_nms(scores, boxes, classes,
                                                   score_threshold=SCORE_THRESHOLD,
                                                   iou_threshold=NMS_IOU_THRESHOLD)

        boxes = tf.reshape(boxes, [-1, 4])
        boxes = boxes.numpy()
        for i in range(boxes.shape[0]):

            # apply the same post-processing steps as in example.py
            box = boxes[i, :]
            shp = orig_image.shape
            orig_h, orig_w = shp[0], shp[1]
            ymin = int(max(1, (box[0] * orig_h)))
            xmin = int(max(1, (box[1] * orig_w)))
            ymax = int(min(orig_h, (box[2] * orig_h)))
            xmax = int(min(orig_w, (box[3] * orig_w)))
            new_box = np.stack([xmin, ymin, xmax, ymax], axis=-1)
            boxes[i, :] = new_box

        boxes = boxes.astype(np.int32)
        boxes[..., 2] = boxes[..., 2] - boxes[..., 0]
        boxes[..., 3] = boxes[..., 3] - boxes[..., 1]
        for score, box, clas in zip(scores.numpy(), boxes, classes.numpy()):

            # class ids are off by 1 compared to our coco labels file
            clas += 1
            results.append({"image_id": image_id,
                            "category_id": int(clas),
                            "bbox": [float(x) for x in list(box)],
                            "score": float(score)})

    return results


predictions = evaluate(interpreter)

with open("predictions.json", "w") as f:
    json.dump(predictions, f, indent=4)

predictions = annotations.loadRes("predictions.json")

os.remove("predictions.json")

cocoeval = COCOeval(annotations, predictions, "bbox")

cocoeval.evaluate()
cocoeval.accumulate()
cocoeval.summarize()
