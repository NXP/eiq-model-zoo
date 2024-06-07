#!/usr/bin/env bash

# SPDX-License-Identifier: MIT
# Copyright 2023-2024 NXP

set -e

python3.8 -m venv env

source ./env/bin/activate 

pip3 install --upgrade tflite2tensorflow


APPVER=v1.20.7
TENSORFLOWVER=2.8.0

wget https://github.com/PINTO0309/tflite2tensorflow/releases/download/${APPVER}/tflite_runtime-${TENSORFLOWVER}-cp38-none-linux_x86_64.whl -O tflite_runtime-${TENSORFLOWVER}-cp38-none-linux_x86_64.whl  \
  && pip3 install --force-reinstall tflite_runtime-${TENSORFLOWVER}-cp38-none-linux_x86_64.whl \
  && rm tflite_runtime-${TENSORFLOWVER}-cp38-none-linux_x86_64.whl

wget https://github.com/PINTO0309/tflite2tensorflow/releases/download/${APPVER}/tensorflow-${TENSORFLOWVER}-cp38-none-linux_x86_64.whl -O tensorflow-${TENSORFLOWVER}-cp38-none-linux_x86_64.whl  \
pip3 install --force-reinstall tensorflow-${TENSORFLOWVER}-cp38-none-linux_x86_64.whl \
  && rm tensorflow-${TENSORFLOWVER}-cp38-none-linux_x86_64.whl

wget https://github.com/PINTO0309/tflite2tensorflow/raw/main/schema/schema.fbs

git clone -b v2.0.8 https://github.com/google/flatbuffers.git
(
cd flatbuffers && mkdir build && cd build
cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
)

pip install -r requirements.txt

wget https://tfhub.dev/intel/lite-model/midas/v2_1_small/1/lite/1?lite-format=tflite -O midas_2_1_small_float32.tflite

tflite2tensorflow \
  --model_path midas_2_1_small_float32.tflite \
  --flatc_path flatbuffers/build/flatc \
  --schema_path schema.fbs \
  --output_pb

tflite2tensorflow \
  --model_path midas_2_1_small_float32.tflite \
  --flatc_path flatbuffers/build/flatc \
  --schema_path schema.fbs \
  --output_integer_quant_tflite \
  --string_formulas_for_normalization 'data / 255'

mv saved_model/model_integer_quant.tflite midas_2_1_small_int8.tflite

# install vela
pip install git+https://github.com/nxp-imx/ethos-u-vela.git@lf-6.1.36-2.1.0

vela --output-dir model_imx93 midas_2_1_small_int8.tflite

#cleanup
deactivate
rm -rf sample_npy saved_model flatbuffers env
rm schema.fbs midas_2_1_small_float32.json