#!/usr/bin/env bash

# SPDX-License-Identifier: MIT
# Copyright 2023 NXP

set -e

wget http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
wget https://github.com/nnsuite/testcases/raw/master/DeepLearningModels/tensorflow-lite/ssd_mobilenet_v2_coco/box_priors.txt
wget https://github.com/nnsuite/testcases/raw/master/DeepLearningModels/tensorflow-lite/ssd_mobilenet_v2_coco/coco_labels_list.txt     

tar xzvf ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz

# tensorflow -> tflite
python3.8 -m venv env_tf
source ./env_tf/bin/activate

pip install --upgrade pip
pip install tensorflow==2.2.0
pip install protobuf==3.19.0
pip install numpy==1.23.5
pip install Pillow==9.5.0

wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip

python3.8 -c "
import tensorflow as tf
import numpy as np

from PIL import Image
from glob import glob
import random


model = tf.saved_model.load('ssdlite_mobilenet_v2_coco_2018_05_09/saved_model/')

m = model.prune('FeatureExtractor/MobilenetV2/MobilenetV2/input:0', ['concat:0', 'concat_1:0'], input_signature=tf.TensorSpec(
    (300,300,3),
    dtype=tf.dtypes.float32,
    name='run'
))

random.seed(1337)

img_list = sorted(glob('val2017/*'))

random.shuffle(img_list)

def _normalize(img):
    mean = [103.53, 116.28, 123.675] 
    std = [57.375, 57.12, 58.395]
    mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3) / 255
    std = np.array(std, dtype=np.float32).reshape(1, 1, 3) / 255
    img = (img - mean) / std
    return img

def representative_data_gen():
  for i in range(100):
    img = Image.open(img_list[i]).convert('RGB')
    img = img.resize((300, 300))
    img = np.array(img, dtype=np.float32) / 255
    img = _normalize(img)
    img = img[None, ...]
    yield [img]


# Convert the model
converter = tf.lite.TFLiteConverter.from_concrete_functions([m])

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
converter.target_spec.supported_types = [tf.int8]

tflite_model = converter.convert()

# Save the model.
with open('ssdlite-mobilenetv2.tflite', 'wb') as f:
  f.write(tflite_model)
"

# install vela
pip install git+https://github.com/nxp-imx/ethos-u-vela.git@lf-6.1.22-2.0.0

vela --output-dir model_imx93 ssdlite-mobilenetv2.tflite

# cleanup 
deactivate
rm ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz val2017.zip
rm -rf val2017 submodel env_tf ssdlite_mobilenet_v2_coco_2018_05_09