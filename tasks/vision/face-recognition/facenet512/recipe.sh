#!/usr/bin/env bash
# Copyright 2022-2025 NXP
# SPDX-License-Identifier: MIT

set -e

python3.8 -m venv env
source ./env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

wget --no-check-certificate https://www.kaggle.com/api/v1/datasets/download/jessicali9530/lfw-dataset -O lfw.zip
unzip lfw.zip

mkdir -p ~/.deepface/weights

python3.8 -c "from deepface.basemodels.Facenet512 import loadModel
import tensorflow as tf
import numpy as np
import random, glob, cv2

random.seed(1337)
model = loadModel()

def representative_dataset():

    files = glob.glob('lfw-deepfunneled/lfw-deepfunneled/*/*jpg')
    random.shuffle(files)

    for filename in files[:100]:

        image = cv2.imread(filename)
        image = cv2.resize(image, (160, 160))
        image = (image[None,...] / 255.0)
    
        yield [image.astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

converter.representative_dataset = representative_dataset

converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.float32

tflite_model = converter.convert()
open('facenet512_uint8.tflite', 'wb').write(tflite_model)
"

deactivate
rm -rf lfw-deepfunneled env lfw.zip
