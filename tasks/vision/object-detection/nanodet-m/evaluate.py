#!/usr/bin/env python3
# Copyright 2022-2023 NXP
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
from example import load_image, generate_center_priors
from example import decode_output, run_inference, COCO_CLASSES, gen_box_colors

OBJECT_DETECTOR_TFLITE = "nanodet_m_0.5x.tflite"
SCORE_THRESHOLD = 0.20
IOU_THRESHOLD = 0.5
INFERENCE_IMG_SIZE = 320

COCO_WEBSITE = "http://images.cocodataset.org"
VAL_IMG_URL = COCO_WEBSITE + "/zips/val2017.zip"
VAL_ANNO_URL = COCO_WEBSITE + "/annotations/annotations_trainval2017.zip"

BOX_COLORS = gen_box_colors()

LABEL_MAP = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
             21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39,
             40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
             57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76,
             77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

print("Downloading COCO validation dataset...")
response = wget.download(VAL_IMG_URL, "val2017.zip")
response = wget.download(VAL_ANNO_URL, "annotations_trainval2017.zip")

with zipfile.ZipFile("val2017.zip", 'r') as zip_ref:
    zip_ref.extractall("coco")

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

        orig_image, img = load_image(image_fn)
        boxes, scores = run_inference(interpreter, img)
        scores, boxes, classes = decode_output(scores, boxes,
                                               SCORE_THRESHOLD, IOU_THRESHOLD)

        shp = orig_image.shape
        boxes = boxes.numpy()
        boxes /= INFERENCE_IMG_SIZE
        boxes *= np.array([shp[1], shp[0], shp[1], shp[0]])

        boxes[..., 2] = boxes[..., 2] - boxes[..., 0]
        boxes[..., 3] = boxes[..., 3] - boxes[..., 1]

        for score, box, clas in zip(scores.numpy(), boxes, classes.numpy()):
            results.append({"image_id": image_id,
                            "category_id": LABEL_MAP[int(clas)],
                            "bbox": list(box),
                            "score": score})

    return results


predictions = evaluate(interpreter)

with open("predictions.json", "w") as f:
    json.dump(predictions, f, indent=4)

predictions = annotations.loadRes("predictions.json")

cocoeval = COCOeval(annotations, predictions, "bbox")

cocoeval.evaluate()
cocoeval.accumulate()
cocoeval.summarize()
