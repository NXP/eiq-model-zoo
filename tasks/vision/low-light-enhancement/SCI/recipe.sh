#!/usr/bin/env bash
# Copyright 2024 NXP
# SPDX-License-Identifier: MIT

set -e

python3.8 -m venv env
source ./env/bin/activate

git clone https://github.com/vis-opt-group/SCI.git SCI_repo
cd SCI_repo 
git apply ../sci.patch

wget https://raw.githubusercontent.com/vis-opt-group/SCI/main/weights/medium.pt

wget https://developer.download.nvidia.com/compute/redist/onnx-graphsurgeon/onnx_graphsurgeon-0.3.9-py2.py3-none-any.whl

pip install --upgrade pip
pip install -r ../requirements.txt

gdown 1G6fi9Kiu7CDnW2Sh7UQ5ikvScRv8Q14F

unzip BrighteningTrain.zip

python3.8 -c "
import onnx2tf
import tensorflow as tf
import numpy as np
import torch
from model import Finetunemodel
import onnx

from PIL import Image
from glob import glob
import random

SIZE = (1920, 1080) # Input resolution (Full HD here)

random.seed(1337)
img_list = sorted(glob('BrighteningTrain/low/*'))
random.shuffle(img_list)

def representative_data_gen():
  for i in range(100):
    img = Image.open(img_list[i]).convert('RGB')
    img = img.resize(SIZE)
    img = np.array(img, dtype=np.float32) / 255
    img = img[None, ...]
    yield [img]

model = Finetunemodel('medium.pt')
model.eval()

inp_size = SIZE
dummy_input = torch.randn(1, 3, inp_size[1], inp_size[0])

# pytorch -> onnx
torch.onnx.export(model,
                  dummy_input,
                  'sci.onnx',
                  verbose=False,
                  opset_version=14)

# onnx -> Tensorflow
onnx2tf.convert(
    input_onnx_file_path='sci.onnx',
    output_folder_path='sci_model',
    output_signaturedefs=True,
    non_verbose=True,
    )

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model('sci_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.float32
converter.target_spec.supported_types = [tf.int8]

tflite_model = converter.convert()

# Save the model.
with open('../sci.tflite', 'wb') as f:
  f.write(tflite_model)
"

deactivate
cd ..
rm -rf SCI_repo