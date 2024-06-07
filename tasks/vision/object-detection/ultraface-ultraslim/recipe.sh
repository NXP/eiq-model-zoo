#!/usr/bin/env bash
# Copyright 2023-2024 NXP
# SPDX-License-Identifier: MIT

set -e

# Clone Ultraface repo and apply patch that modifies the post processing steps.
git clone https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB
(
cd Ultra-Light-Fast-Generic-Face-Detector-1MB || exit
git checkout dffdddd
git apply ../ultraslim.patch

# Download widerface test set for quantization calibration
wget https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_test.zip
unzip WIDER_test.zip
rm WIDER_test.zip

# Install requirements in virtual env
python3.8 -m venv env
source ./env/bin/activate
pip install --upgrade pip
pip install -r ../requirements_tf.txt

# Generate tensorflow weights
(
cd tf
python3.8 convert_tensorflow.py --net_type slim
)

deactivate
python3.9 -m venv env_tflite
source ./env_tflite/bin/activate
pip install --upgrade pip
pip install "tensorflow==2.14.0"
pip install opencv-python

# Convert model to tflite
python3.9 -c "
import cv2, numpy as np, tensorflow as tf, glob, random
random.seed(1337)

SAVE_MODEL_DIR = 'tf/export_models/slim_ultraslim/'
OUTPUT_TF_FILE_NAME = '../ultraface_ultraslim_uint8_float32.tflite'
OUTPUT_TF_FILE_NAME_INT8 = '../ultraface_ultraslim_int8.tflite'
IMAGES_PATH = 'WIDER_test/images/*/*jpg'

NIMAGES_REPRESENTATIVE_DATASET = 100

def representative_dataset():
    files = glob.glob(IMAGES_PATH)
    random.shuffle(files)

    for filename in files[:NIMAGES_REPRESENTATIVE_DATASET]:

        image = cv2.imread(filename)
        image = cv2.resize(image, (128, 128))
        image = (image[None,...] - 127.0)/128.0
    
        yield [image.astype(np.float32)]


def main():
    # converter = tf.lite.TFLiteConverter.from_saved_model(SAVE_MODEL_DIR)
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    # converter.representative_dataset = representative_dataset

    # converter.inference_input_type = tf.uint8 
    # converter.inference_output_type = tf.float32 

    # tflite_model = converter.convert()
    # open(OUTPUT_TF_FILE_NAME, 'wb').write(tflite_model)

    converter = tf.lite.TFLiteConverter.from_saved_model(SAVE_MODEL_DIR)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    converter.representative_dataset = representative_dataset

    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8 

    tflite_model = converter.convert()
    open(OUTPUT_TF_FILE_NAME_INT8, 'wb').write(tflite_model)

if __name__ == '__main__':
    main()
"

)

rm -rf Ultra-Light-Fast-Generic-Face-Detector-1MB
