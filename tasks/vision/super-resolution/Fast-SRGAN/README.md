# Fast-SRGAN

## Introduction

Fast-SRGAN is a lightweight Super-Resolution model.
It can upscale small images to 4x their size with minimal loss in detail.

NB: The model implemented here is Fast-SRGAN x4 Bicubic specifically. Hence, the model will provide best results for bicubically downscaled images.

## Model Information

Information   | Value
---           | ---
Input shape   | RGB image
Input example | <img src="example_input.png" width=128px> (Image source: Urban100 dataset [1])
Output shape  | Tensor of size 4 x Original Image size.
Output example | <img src="example_output.png" width=512px>
FLOPS | 460M
Number of parameters | 168K
File size (int8) | 240K
Source framework | Tensorflow / Keras
Target platform | MPUs

## Version and changelog

Initial release of quantized int8 model.

## Tested configurations

The int8 model has been tested on i.MX 8MP and i.MX 93 using benchmark-model (see [i.MX Machine Learning User Guide](https://www.nxp.com/docs/en/user-guide/IMX-MACHINE-LEARNING-UG.pdf)).

## Training and evaluation

The model has been trained and evaluated on the DIV2k dataset. It achives comparable qualitative result to the model it aims to speed up (SRGAN) while being noticeably faster according to [the source of the model](https://github.com/HasnainRaz/Fast-SRGAN/tree/master?tab=readme-ov-file). The model the source uses has 6 inverted residual blocks and 32 filters in every layers of its generator.

## Conversion/Quantization

The recipe script downloads the already quantised tflite file from [this](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/171_Fast-SRGAN) repo.

## Use case and limitations

This model can be used for Super-Resolution on small sized images.
The original model was trained on 128x128 images but it can still be used with images of different sizes.

## Download and run

To create the TensorFlow Lite model fully quantized in int8 with float32 input and float32 output, follow the top-level README instructions to install Docker and build the Docker image, then run the following command: 

    docker run --rm -v "$PWD:/workspace" nxp-model-zoo recipe.sh

The TensorFlow Lite model file for i.MX 8M Plus and i.MX 93 is `fsrgan.tflite`.

An example of how to use the model is in `example.py`.

## Origin

Model implementation: https://github.com/HasnainRaz/Fast-SRGAN/tree/master?tab=readme-ov-file

[1] Urban100 dataset:  Huang et al. Single Image Super-Resolution From Transformed Self-Exemplars. in CVPR 2015

