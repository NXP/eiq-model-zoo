#!/usr/bin/env bash
# Copyright 2023 NXP
# SPDX-License-Identifier: MIT

set -e

wget https://github.com/Ascend-Research/HeadPoseEstimation-WHENet/raw/a0d7bdfb5e2ac97ae6b0ae3eef79fdcf4075ab82/WHENet.h5 -O WHENet.h5

python3.8 -m venv env
source ./env/bin/activate
pip install --upgrade pip

pip install -r requirements.txt

# Convert model to tflite
python3.8 -c "
import cv2, numpy as np, tensorflow as tf, glob

import numpy as np
import tensorflow_datasets as tfds
from utils import get_whenet

NIMAGES_REPRESENTATIVE_DATASET = 100
read_config = tfds.ReadConfig()
read_config.shuffle_seed = 1337
ds = tfds.load('the300w_lp', split=f'train[:{NIMAGES_REPRESENTATIVE_DATASET}]', shuffle_files=True, read_config=read_config)

model = get_whenet()

model.load_weights('WHENet.h5')

model.summary()

OUTPUT_TF_FILE_NAME = 'whenet.tflite'

def representative_dataset():

    for ex in ds:

        image = ex['image'].numpy()
        image = cv2.resize(image, (224, 224))

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image = image[None,...]/255
        image = (image - mean) / std
    
        yield [image.astype(np.float32)]


def main():

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    converter.representative_dataset = representative_dataset

    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8 

    tflite_model = converter.convert()
    open(OUTPUT_TF_FILE_NAME, 'wb').write(tflite_model)

if __name__ == '__main__':
    main()
"

# install vela
pip install git+https://github.com/nxp-imx/ethos-u-vela.git@lf-6.1.22-2.0.0
pip install numpy==1.23.5
vela --output-dir model_imx93 whenet.tflite
