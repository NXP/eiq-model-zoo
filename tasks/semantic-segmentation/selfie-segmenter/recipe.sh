#!/usr/bin/env bash
# Copyright 2023 NXP

set -e

# fetch calibration images from COCO dataset
wget http://images.cocodataset.org/zips/val2017.zip -O val2017.zip
unzip val2017.zip

mkdir -p calib_data
for file in $(cat data/calib_images_list.txt); do mv "$file" calib_data; done
rm -rf val2017.zip val2017

# Select which model to use (General = 0, Landscape = 1)
[[ -z "${USE_LANDSCAPE}" ]] && MODEL_VERSION=0 || MODEL_VERSION=${USE_LANDSCAPE}
if [[ $MODEL_VERSION -eq 0 ]]
  then
    MODEL="selfie_segmenter"
    SHAPE="(256, 256)"
  else
    MODEL="selfie_segmenter_landscape"
    SHAPE="(256, 144)"
fi

# Download Selfie Segmenter model from MediaPipe's webpage
mkdir -p tmp && cd tmp
wget https://storage.googleapis.com/mediapipe-models/image_segmenter/${MODEL}/float16/latest/${MODEL}.tflite
cd ../ && mv tmp/${MODEL}.tflite .
rm -rf tmp

# Create folder for model conversion
mkdir tmp
(
    cd tmp
    
    # Clone flatbuffers repo and build it
    git clone -b v2.0.8 https://github.com/google/flatbuffers.git
    cd flatbuffers && mkdir build && cd build
    cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release ..
    make -j"$(nproc)"
    
    # Download schema.fbs
    cd ../../
    wget https://github.com/PINTO0309/tflite2tensorflow/raw/main/schema/schema.fbs
    
    # MediaPipe *.tflite -> TensorFlow (*.pb)
    python3 -m venv env_tflite2tf
    source ./env_tflite2tf/bin/activate
    pip install --upgrade pip
    pip install pandas
    pip install --upgrade git+https://github.com/PINTO0309/tflite2tensorflow
    
    APPVER=v1.20.7
    TENSORFLOWVER=2.8.0
    
    wget https://github.com/PINTO0309/tflite2tensorflow/releases/download/${APPVER}/tflite_runtime-${TENSORFLOWVER}-cp38-none-linux_x86_64.whl &&
    pip3 install --force-reinstall tflite_runtime-${TENSORFLOWVER}-cp38-none-linux_x86_64.whl &&
    rm tflite_runtime-${TENSORFLOWVER}-cp38-none-linux_x86_64.whl
    
    wget https://github.com/PINTO0309/tflite2tensorflow/releases/download/${APPVER}/tensorflow-${TENSORFLOWVER}-cp38-none-linux_x86_64.whl &&
    pip3 install --force-reinstall tensorflow-${TENSORFLOWVER}-cp38-none-linux_x86_64.whl &&
    rm tensorflow-${TENSORFLOWVER}-cp38-none-linux_x86_64.whl
    
    # Fix protobuf version
    pip install protobuf==3.20
    
    # Convert MediaPipe Model to TF (*.pb)
    cd ../
    tflite2tensorflow --model_path ${MODEL}.tflite --flatc_path ./tmp/flatbuffers/build/flatc --schema_path ./tmp/schema.fbs --output_pb
    deactivate
)

# Remove tmp folder
rm -rf tmp
rm ${MODEL}.json

# Create folder for model quantization
mkdir tmp
(
    # TensorFlow (*.pb) -> TFLite (*.tflite)
    cd tmp
    python3 -m venv env_tf
    source ./env_tf/bin/activate
    
    pip install --upgrade pip
    pip install wheel
    pip install -r ../requirements_tf.txt
    
    python -c "
import random
import numpy as np
import tensorflow as tf
from PIL import Image
from glob import glob

random.seed(1337)

list_images = sorted(glob('../calib_data/*'))
random.shuffle(list_images)

def normalize_input(input_data):
    normalized_data = np.ascontiguousarray(np.asarray(input_data) / 255.0).astype('float32')
    normalized_data = normalized_data[None, ...]
    return normalized_data

def generate_representative_data():
  for i in range(40):
    input_data = Image.open(list_images[i]).convert('RGB')
    input_data = input_data.resize(${SHAPE})
    input_data = normalize_input(input_data)
    yield [input_data]

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model('../saved_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.representative_dataset = generate_representative_data
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32
tflite_model = converter.convert()

# Save the model
with open('../${MODEL}_int8.tflite', 'wb') as f:
  f.write(tflite_model)
    "
        
    # install vela
    pip install git+https://github.com/nxp-imx/ethos-u-vela.git@lf-6.1.1-1.0.0
    vela --output-dir ../model_imx93 ../${MODEL}_int8.tflite
    deactivate
)

# Remove tmp folder
rm -rf saved_model
rm -rf tmp
rm -rf calib_data
