#!/usr/bin/env bash
# Copyright 2022-2023 NXP

set -e

if [ -z "$KAGGLE_USERNAME" ] || [ -z "$KAGGLE_KEY" ]
then
      echo "ERROR: This recipe script requires a Kaggle API key. 
Follow the instructions in README.md to create an API key and add it to KAGGLE_USERNAME and KAGGLE_KEY environment variables."
      exit 1
fi

python3 -m venv env
source ./env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt


# download dataset for calibration
kaggle competitions download -c challenges-in-representation-learning-facial-expression-recognition-challenge
unzip challenges-in-representation-learning-facial-expression-recognition-challenge.zip
tar xzvf fer2013.tar.gz

mkdir -p ~/.deepface/weights

python -c "from deepface.extendedmodels.Emotion import loadModel
import tensorflow as tf
import numpy as np
import random, glob, cv2

random.seed(1337)
model = loadModel()

OUTPUT_TF_FILE_NAME = 'emotion_uint8_float32.tflite'
IMAGES_PATH = 'test/*/*jpg'
NIMAGES_REPRESENTATIVE_DATASET = 100

with open('fer2013/fer2013.csv', 'r') as f:
    lines = f.readlines()

lines = [l.strip().split(',') for l in lines[1:]]
lines = [(int(l[0]), np.array(l[1].split(' ')).reshape((48,48,1)).astype(np.float32)) for l in lines if l[2] == 'PrivateTest']


def representative_dataset():

    random.shuffle(lines)

    for line in lines[:NIMAGES_REPRESENTATIVE_DATASET]:

        image = line[1]
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
pip install git+https://github.com/nxp-imx/ethos-u-vela.git@lf-6.1.1-1.0.0

vela --output-dir model_imx93 emotion_uint8_float32.tflite

deactivate
rm -rf fer2013 fer2013.tar.gz env challenges-in-representation-learning-facial-expression-recognition-challenge.zip example_submission.csv icml_face_data.csv test.csv train.csv
