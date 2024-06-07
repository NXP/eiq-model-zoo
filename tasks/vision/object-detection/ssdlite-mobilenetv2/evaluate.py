#!/usr/bin/env python3
# Copyright 2023-2024 NXP
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
from example import load_image
from example import decode_output, run_inference, gen_box_colors

OBJECT_DETECTOR_TFLITE = "ssdlite-mobilenetv2.tflite"
SCORE_THRESHOLD = 0.20
IOU_THRESHOLD = 0.5
INFERENCE_IMG_SIZE = 300

COCO_WEBSITE = "http://images.cocodataset.org"
VAL_IMG_URL = COCO_WEBSITE + "/zips/val2017.zip"
VAL_ANNO_URL = COCO_WEBSITE + "/annotations/annotations_trainval2017.zip"

BOX_COLORS = gen_box_colors()

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
                            "category_id": int(clas),
                            "bbox": [float(x) for x in list(box)],
                            "score": float(score)})

    return results


predictions = evaluate(interpreter)

with open("predictions.json", "w") as f:
    json.dump(predictions, f, indent=4)

predictions = annotations.loadRes("predictions.json")

cocoeval = COCOeval(annotations, predictions, "bbox")

cocoeval.evaluate()
cocoeval.accumulate()
cocoeval.summarize()
