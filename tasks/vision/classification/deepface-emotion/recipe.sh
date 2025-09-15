#!/usr/bin/env bash
# Copyright 2022-2025 NXP
# SPDX-License-Identifier: MIT

set -e

python3.8 -m venv env
source ./env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt


# download dataset for calibration
wget --no-check-certificate -O fer2013.zip https://www.kaggle.com/api/v1/datasets/download/msambare/fer2013
unzip fer2013.zip


mkdir -p ~/.deepface/weights

python3.8 -c "from deepface.extendedmodels.Emotion import loadModel
import tensorflow as tf
import numpy as np
import random, glob, cv2

random.seed(1337)
model = loadModel()

OUTPUT_TF_FILE_NAME = 'emotion_uint8_float32.tflite'
IMAGES_PATH = 'test/*/*jpg'
NIMAGES_REPRESENTATIVE_DATASET = 100

files = glob.glob(IMAGES_PATH)
random.shuffle(files)

def representative_dataset():

    for filename in files[:100]:

        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)[..., None]
        image = (image[None, ...] /255.0)
    
        yield [image.astype(np.float32)]


converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

converter.representative_dataset = representative_dataset

converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.float32  

tflite_model = converter.convert()
open(OUTPUT_TF_FILE_NAME, 'wb').write(tflite_model)
"

# install vela
pip install git+https://github.com/nxp-imx/ethos-u-vela.git@lf-6.1.22-2.0.0

vela --output-dir model_imx93 emotion_uint8_float32.tflite

deactivate
rm -rf fer2013 fer2013.tar.gz env challenges-in-representation-learning-facial-expression-recognition-challenge.zip example_submission.csv icml_face_data.csv test.csv train.csv
