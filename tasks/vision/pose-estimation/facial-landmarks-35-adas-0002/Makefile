# Copyright 2024 NXP
# SPDX-License-Identifier: MIT

all : model_full_integer_quant.tflite

model_full_integer_quant.tflite: intel/facial-landmarks-35-adas-0002/FP16-INT8/facial-landmarks-35-adas-0002.bin env/bin/activate calib.npy
	. ./env/bin/activate ; \
	openvino2tensorflow --model_path intel/facial-landmarks-35-adas-0002/FP32/facial-landmarks-35-adas-0002.xml --output_saved_model --output_full_integer_quant_tflite  --output_integer_quant_type uint8 --load_dest_file_path_for_the_calib_npy calib.npy --calib_ds_type numpy --string_formulas_for_normalization 'data'
	mv saved_model/model_full_integer_quant.tflite model_full_integer_quant.tflite

intel/facial-landmarks-35-adas-0002/FP16-INT8/facial-landmarks-35-adas-0002.bin : env/bin/activate
	. ./env/bin/activate ; \
	omz_downloader --name facial-landmarks-35-adas-0002

calib.npy : lfw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg env/bin/activate
	. ./env/bin/activate ; \
	python calibration_data.py

env/bin/activate :
	python3.9 -m venv env
	. ./env/bin/activate ; \
	pip install --upgrade pip ; \
	pip install -r requirements.txt ; \
	git clone https://github.com/openvinotoolkit/open_model_zoo ; \
	cd open_model_zoo/tools/model_tools/ ; \
	pip install .

lfw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg : lfw.tgz
	tar xzf lfw.tgz
	touch lfw/*/*.jpg

lfw.tgz :
	wget http://vis-www.cs.umass.edu/lfw/lfw.tgz -O lfw.tgz

clean :
	rm -rf open_model_zoo lfw lfw.tgz env model_full_integer_quant.tflite calib.npy intel saved_model