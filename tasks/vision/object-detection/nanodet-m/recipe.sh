#!/usr/bin/env bash
# Copyright 2022-2023 NXP
# SPDX-License-Identifier: MIT

set -e

git clone https://github.com/RangiLyu/nanodet
(
cd nanodet || exit
git checkout 0f4d8f11443
git apply ../nanodet.patch

# pytorch -> onnx
python3.8 -m venv env_pt2onnx
source ./env_pt2onnx/bin/activate

pip install --upgrade pip
pip install wheel
pip install -r ../requirements_pytorch_to_onnx.txt
pip install gdown

gdown 1rMHkD30jacjRpslmQja5jls86xd0YssR

pip install -e .

python3.8 tools/export_onnx.py --cfg_path config/legacy_v0.x_configs/nanodet-m-0.5x.yml --model_path ./nanodet_m_0.5x.ckpt --out_path nanodet_m_0.5x.onnx

deactivate


# onnx -> openvino
python3.8 -m venv env_onnx2ov
source ./env_onnx2ov/bin/activate

pip install --upgrade pip
pip install wheel
pip install -r ../requirements_onnx_to_openvino.txt 

mo --input_model nanodet_m_0.5x.onnx

deactivate

# openvino -> tensorflow
python3.8 -m venv env_ov2tf
source ./env_ov2tf/bin/activate

pip install --upgrade pip
pip install wheel 
pip install -r ../requirements_openvino_to_tf.txt 

openvino2tensorflow --model_path nanodet_m_0.5x.xml --model_output_path nanodet_m_0.5x.pb --output_pb --output_saved_model

deactivate

# tensorflow -> tflite
python3.8 -m venv env_tf
source ./env_tf/bin/activate

pip install --upgrade pip
pip install wheel 
pip install -r ../requirements_tf.txt 

wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip

python3.8 -c "
import tensorflow as tf
import numpy as np

from PIL import Image
from glob import glob
import random

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
    img = img.resize((320, 320))
    img = np.array(img, dtype=np.float32) / 255
    img = _normalize(img)
    img = img[None, ...]
    yield [img]


# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model('nanodet_m_0.5x.pb')

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
converter.target_spec.supported_types = [tf.int8]

tflite_model = converter.convert()

# Save the model.
with open('../nanodet_m_0.5x.tflite', 'wb') as f:
  f.write(tflite_model)
"
)

# cleanup
rm -rf nanodet
