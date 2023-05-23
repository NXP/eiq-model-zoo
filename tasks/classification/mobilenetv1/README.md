# MobileNet V1

## Introduction

[MobileNet V1](https://github.com/serengil/deepface) [1] is a family of lightweight convolutional neural networks for image classification.
We use the weights from the implementation found in [TensorFlow Model Garden](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md).

Our base recipe script offers the smallest model (width multiplier 0.25, input resolution 128x128) by default.

## Model Information

Information   | Value
---           | ---
Input shape   | Color image (128, 128, 3)
Input example | <img src="example_input.jpg"> ([Public domain image](https://commons.wikimedia.org/wiki/File:Traffic_light_green_Drammen_(2).jpg))
Output shape  | Vector of probabilities shape (1, 1001). Labels can be found in `labels.py`.
Output example | Output tensor: [[0., 0.,  0., 58, ... 0., 0., 0. ]]
FLOPS | 28 MOPS
Number of parameters | 468k
Source framework | Tensorflow/Keras
Target platform | MPUs, MCUs

## Version and changelog

Initial release of quantized int8 model.

## Tested configurations

The quantized int8 models have been tested on i.MX 8M Plus and i.MX 93 using benchmark-model (see [i.MX Machine Learning User Guide](https://www.nxp.com/docs/en/user-guide/IMX-MACHINE-LEARNING-UG.pdf)), and on i.MX RT1170 and i.MX RT1050 using TensorFlow Lite Micro.

## Training and evaluation

The model has been trained and evaluated on the ImageNet dataset [2]. It achieved a score of 39.5% Top-1 accuracy on the test set according to [the source of the model](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md).

The original training procedure is detailed [here](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1_train.py).

## Conversion/Quantization

The model is downloaded in an archive which contains the quantized model and original (*.pb) TensorFlow model. See [the source of the model](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md) for information on the quantization procedure that was used.

## Use case and limitations

This model can be used for simple object recognition applications. Since its accuracy is relatively low, it should not be trusted for critical scenarios.
If necessary, a more accurate model can be obtained by modifying the `recipe.sh` script and selecting a larger model scale and/or input resolution. A larger model will be slower, however.

## Performance

Here are performance figures evaluated on i.MX 8MP and i.MX 93 using BSP LF6.1.1_1.0.0::

Model   | Average latency | Platform        | Accelerator       | Command
---     | ---             | ---             | ---               | ---
Int8    | 6.53ms          | i.MX 8M Plus    |   CPU (1 thread)  | /usr/bin/tensorflow-lite-2.10.0/examples/benchmark_model --graph=mobilenet_v1_0.25_128_quant.tflite
Int8    | 2.44ms          | i.MX 8M Plus    |   CPU (4 threads) | /usr/bin/tensorflow-lite-2.10.0/examples/benchmark_model --graph=mobilenet_v1_0.25_128_quant.tflite --num_threads=4
Int8    | 0.84ms          | i.MX 8M Plus    |   NPU             | /usr/bin/tensorflow-lite-2.10.0/examples/benchmark_model --graph=mobilenet_v1_0.25_128_quant.tflite --external_delegate_path=/usr/lib/libvx_delegate.so
Int8    | 3.63ms          | i.MX 93         |   CPU (1 thread)  | /usr/bin/tensorflow-lite-2.10.0/examples/benchmark_model --graph=mobilenet_v1_0.25_128_quant.tflite
Int8    | 2.41ms          | i.MX 93         |   CPU (2 threads) | /usr/bin/tensorflow-lite-2.10.0/examples/benchmark_model --graph=mobilenet_v1_0.25_128_quant.tflite --num_threads=2
Int8    | 0.37ms          | i.MX 93         |   NPU             | /usr/bin/tensorflow-lite-2.10.0/examples/benchmark_model --graph=mobilenet_v1_0.25_128_quant_vela.tflite --external_delegate_path=/usr/lib/libethosu_delegate.so
Int8    | 47ms            | i.MX RT1170     |   CPU             | Tested with TensorFlow Lite Micro
Int8    | 70ms            | i.MX RT1050     |   CPU             | Tested with TensorFlow Lite Micro

## Download and run

To create the TFLite model fully quantized in int8 with int8 input and float32 output, run `bash recipe.sh`.

The TFLite model file for i.MX 8M Plus, i.MX RT 1170 and i.MX RT1050 is `mobilenet_v1_0.25_128_quant.tflite`. The file for i.MX 93 is output in the `model_imx93` directory.

An example of how to use the model is in `example.py`. The output labels are listed in `labels.py`.

## Origin

Model implementation: https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md

[1] Howard, Andrew G., et al. "Mobilenets: Efficient convolutional neural networks for mobile vision applications." arXiv preprint arXiv:1704.04861 (2017).
[2] Deng, Jia, et al. "Imagenet: A large-scale hierarchical image database." 2009 IEEE conference on computer vision and pattern recognition. Ieee, 2009.
