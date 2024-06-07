#!/usr/bin/env bash
# Copyright 2024 NXP
# SPDX-License-Identifier: MIT

set -e

python3.8 -m venv env
source ./env/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

git clone --depth 1 https://github.com/tensorflow/models

cd models || exit
git apply ../centernet.patch
cd ..

(
  cd models/research/
  protoc object_detection/protos/*.proto --python_out=.
  cp object_detection/packages/tf2/setup.py .
  python -m pip install .
)

wget https://raw.githubusercontent.com/amikelive/coco-labels/master/coco-labels-2014_2017.txt

wget http://download.tensorflow.org/models/object_detection/tf2/20210210/centernet_mobilenetv2fpn_512x512_coco17_od.tar.gz
tar -xf centernet_mobilenetv2fpn_512x512_coco17_od.tar.gz
rm centernet_mobilenetv2fpn_512x512_coco17_od.tar.gz*

# change the height and width field to use another resolution.
python models/research/object_detection/export_tflite_graph_tf2.py \
  --pipeline_config_path=centernet_mobilenetv2_fpn_od/pipeline.config \
  --trained_checkpoint_dir=centernet_mobilenetv2_fpn_od/checkpoint \
  --output_directory=centernet_mobilenetv2_fpn_od/tflite \
  --centernet_include_keypoints=false \
  --max_detections=10 \
  --config_override=" \
    model{ \
      center_net { \
        image_resizer { \
          fixed_shape_resizer { \
            height: 512 \
            width: 512 \
          } \
        } \
        feature_extractor { \
          use_separable_conv: true \
        } \
      } \
    }"

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

# change the two values in the resize call (width, height) 
# to use another resolution.
def representative_data_gen():
  for i in range(100):
    img = Image.open(img_list[i]).convert('RGB')
    img = img.resize((512, 512))
    img = np.array(img, dtype=np.float32) 
    img = img[None, ...]
    yield [img]


# convert the model
converter = tf.lite.TFLiteConverter.from_saved_model('centernet_mobilenetv2_fpn_od/tflite/saved_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.float32
converter.target_spec.supported_types = [tf.int8]

tflite_model = converter.convert()

# save the model.
with open('centernet.tflite', 'wb') as f:
  f.write(tflite_model)
"

deactivate
rm -rf centernet_mobilenetv2_fpn_od models val2017.zip val2017