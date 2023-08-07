#!/usr/bin/env bash
# Copyright 2022-2023 NXP

set -e

# Clone Ultraface repo and apply patch that modifies the post processing steps.
git clone https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB
(
cd Ultra-Light-Fast-Generic-Face-Detector-1MB || exit
git checkout dffdddd
git apply ../slim.patch

# Download widerface test set for quantization calibration
wget https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_test.zip
unzip WIDER_test.zip
rm WIDER_test.zip

# Install requirements in virtual env
python3 -m venv env
source ./env/bin/activate
pip install --upgrade pip
pip install -r ../requirements_tf.txt

# Generate tensorflow weights
(
cd tf
python convert_tensorflow.py --net_type slim
)

# Convert model to tflite
python -c "
import cv2, numpy as np, tensorflow as tf, glob, random
random.seed(1337)

SAVE_MODEL_DIR = 'tf/export_models/slim/'
OUTPUT_TF_FILE_NAME = '../ultraface_slim_uint8_float32.tflite'
OUTPUT_TF_FILE_NAME_INT8 = '../ultraface_slim_int8.tflite'
IMAGES_PATH = 'WIDER_test/images/*/*jpg'

NIMAGES_REPRESENTATIVE_DATASET = 100

def representative_dataset():
    files = glob.glob(IMAGES_PATH)
    random.shuffle(files)

    for filename in files[:NIMAGES_REPRESENTATIVE_DATASET]:

        image = cv2.imread(filename)
        image = cv2.resize(image, (320, 240))
        image = (image[None,...] - 127.0)/128.0
    
        yield [image.astype(np.float32)]


def main():
    converter = tf.lite.TFLiteConverter.from_saved_model(SAVE_MODEL_DIR)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    converter.representative_dataset = representative_dataset

    converter.inference_input_type = tf.uint8 
    converter.inference_output_type = tf.float32 

    tflite_model = converter.convert()
    open(OUTPUT_TF_FILE_NAME, 'wb').write(tflite_model)

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

# install vela
pip install git+https://github.com/nxp-imx/ethos-u-vela.git@lf-6.1.22-2.0.0

vela --output-dir ../model_imx93 ../ultraface_slim_uint8_float32.tflite
)

rm -rf Ultra-Light-Fast-Generic-Face-Detector-1MB