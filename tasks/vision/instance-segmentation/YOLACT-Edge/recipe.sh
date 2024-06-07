#!/usr/bin/env bash
# Copyright 2024 NXP
# SPDX-License-Identifier: MIT

set -e

# pytorch -> onnx
python3.8 -m venv env
source ./env/bin/activate

wget https://github.com/PINTO0309/yolact_edge_onnx_tensorrt_myriad/releases/download/1.0.4/yolact_edge_mobilenetv2_54_800000.onnx

wget https://developer.download.nvidia.com/compute/redist/onnx-graphsurgeon/onnx_graphsurgeon-0.3.9-py2.py3-none-any.whl

pip install --upgrade pip
pip install -r requirements.txt
pip install onnx_graphsurgeon-0.3.9-py2.py3-none-any.whl

wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip

python3.8 -c "
import onnx2tf
import tensorflow as tf
import numpy as np

from PIL import Image
from glob import glob
import random

random.seed(1337)

img_list = sorted(glob('val2017/*'))

random.shuffle(img_list)

def representative_data_gen():
  for i in range(100):
    img = Image.open(img_list[i]).convert('RGB')
    img = img.resize((550, 550))
    img = np.array(img, dtype=np.float32) 
    img = img[None, ...]
    yield [img]

onnx2tf.convert(
    input_onnx_file_path='yolact_edge_mobilenetv2_54_800000.onnx',
    output_folder_path='yolact_edge_mobilenet',
    copy_onnx_input_output_names_to_tflite=True,
    output_signaturedefs=True,
    non_verbose=True,
)
 
# Convert the model

converter = tf.lite.TFLiteConverter.from_saved_model('yolact_edge_mobilenet')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.float32
converter.target_spec.supported_types = [tf.int8]

tflite_model = converter.convert()

# Save the model.
with open('YOLACT-Edge.tflite', 'wb') as f:
  f.write(tflite_model)
"

deactivate
rm yolact_edge_mobilenetv2_54_800000.onnx
rm onnx_graphsurgeon-0.3.9-py2.py3-none-any.whl
rm val2017.zip 
rm -rf yolact_edge_mobilenet
 