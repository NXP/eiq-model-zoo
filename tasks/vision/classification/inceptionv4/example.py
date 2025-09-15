#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright 2024 NXP

import argparse
import numpy as np
import tensorflow as tf
from PIL import Image

from labels import IMAGENET_LABELS


def inference(modelPath, inputPath):
    img = Image.open(inputPath)
    img = img.resize((299, 299))

    img = np.array(img, dtype=np.int64)
    tensor = img.astype(np.uint8)


    tensor = np.expand_dims(tensor, axis=0)

    interpreter = tf.lite.Interpreter(model_path=modelPath)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]["index"], tensor)
    interpreter.invoke()

    pred_y = interpreter.get_tensor(output_details[0]["index"])

    print(IMAGENET_LABELS[np.argmax(pred_y)-1])


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-i",
        "--input",
        action="store",
        dest="inputSamplePath",
        default="example_input.jpg",
        help="Path to input image",
    )
    argparser.add_argument(
        "-m",
        "--model",
        action="store",
        dest="modelPath",
        default="inceptionv4_quant_int8.tflite",
        help="Path to TFLite model.",
    )

    args = argparser.parse_args()

    inference(args.modelPath, args.inputSamplePath)
