# Ultraface slim

## Introduction

["Ultraface"](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB) is a lightweight face detection model designed for edge computing devices.
It regresses bounding boxes (4 coordinates) and a confidence score for each box. The bounding box decoding and non-maximum suppression steps are included in the model.

NB: This is only a face detection model. It does not perform face recognition/identification.

## Model Information

Information   | Value
---           | ---
Input shape   | RGB image (320, 240, 3)
Input example | <img src="example_input.jpg" width=320px> (Image source: NASA, Public domain)
Output shape  | Tensor of size (100, 6) containing a maximum of 100 detected faces.
Output example | <img src="example_output.jpg" width=320px>
FLOPS | 168,707,432
Number of parameters | 264,732
File size (float32) | 1.04MB
File size (int8) | 403KB
Source framework | Pytorch
Target platform | MPUs, MCUs

## Version and changelog

Initial release of float32 and quantized int8 model.

## Tested configurations

The int8 model has been tested on i.MX 8MP and i.MX 93 using benchmark-model (see [i.MX Machine Learning User Guide](https://www.nxp.com/docs/en/user-guide/IMX-MACHINE-LEARNING-UG.pdf)) and on i.MX RT 1170 and i.MX RT 1050 using TensorFlow Lite Micro.

## Training and evaluation

The model has been trained and evaluated on the Widerface dataset. It achieved scores of 0.77/0.671/0.395 on the easy/medium/hard sets, according to [the source of the model](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB).

## Conversion/Quantization

The original model is converted from PyTorch to TensorFlow, and then to TensorFlow Lite.

The conversion script performs this conversion and outputs the float32 model and int8 quantized model. 100 random images from the Widerface test dataset are used as calibration for the quantization.

## Use case and limitations

This model can be used for very fast face detection on 320x240 pixel images.

## Performance

Here are performance figures evaluated on supported products (BSP LF6.1.1_1.0.0):

Model   | Average latency | Platform     | Accelerator | Command
---     | ---             | ---          | ---         | ---
Int8    | 54.44ms         | i.MX 8MP     |   CPU (1 thread)  | /usr/bin/tensorflow-lite-2.10.0/examples/benchmark_model --graph=ultraface_slim_uint8_float32.tflite
Int8    | 29.34ms         | i.MX 8MP     |   CPU (4 threads) | /usr/bin/tensorflow-lite-2.10.0/examples/benchmark_model --graph=ultraface_slim_uint8_float32.tflite --num_threads=4
Int8    | 5.23ms          | i.MX 8MP     |   NPU             | /usr/bin/tensorflow-lite-2.10.0/examples/benchmark_model --graph=ultraface_slim_uint8_float32.tflite --external_delegate_path=/usr/lib/libvx_delegate.so
Int8    | 39.38ms         | i.MX 93      |   CPU (1 thread)  | /usr/bin/tensorflow-lite-2.10.0/examples/benchmark_model --graph=ultraface_slim_uint8_float32.tflite
Int8    | 32.50ms         | i.MX 93      |   CPU (2 threads) | /usr/bin/tensorflow-lite-2.10.0/examples/benchmark_model --graph=ultraface_slim_uint8_float32.tflite --num_threads=2
Int8    | 6.66ms          | i.MX 93      |   NPU             | /usr/bin/tensorflow-lite-2.10.0/examples/benchmark_model --graph=ultraface_slim_uint8_float32_vela.tflite --external_delegate_path=/usr/lib/libethosu_delegate.so
Int8    | 566ms           | i.MX RT 1170 |   CPU             | Used with TFLite micro
Int8    | 1788ms          | i.MX RT 1050 |   CPU             | Used with TFLite micro

## Download and run

To create the TensorFlow Lite model fully quantized in int8 with int8 input and float32 output, run:

    bash recipe.sh

The TensorFlow Lite model file for i.MX 8M Plus and i.MX RT 1170 is `ultraface_slim_uint8_float32.tflite`. The file for i.MX RT 1170 and 1050 is `ultraface_slim_int8.tflite`. The file for i.MX 93 is in the `model_imx93` directory.

An example of how to use the model is in `example.py`.

## Origin

Model implementation: https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB

[1] WIDERface dataset: Yang, Shuo, et al. "Wider face: A face detection benchmark." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
