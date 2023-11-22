# Tiny ResNet

## Introduction

Tiny ResNet is small image classification model. The model is trained on small images (32 x 32) from CIFAR10 [1] dataset, distributed into 10 different classes.

We use model trained by [MLCommons](https://mlcommons.org/en/) available on their [GitHub page](https://github.com/mlcommons/tiny/tree/master/benchmark/training/image_classification) [2].

## Model Information

| Information          | Value                                                            |
|----------------------|------------------------------------------------------------------|
| Input shape          | RGB Color image (32, 32, 3)                                      |
| Input example        | <img src="example_input.jpg"> (Image from CIFAR10 dataset [1])   |
| Output shape         | Vector of probabilities shape (1, 10).                           |
| Output example       | Output tensor: [[0. 0. 0. 0.99609375 0. 0.00390625 0. 0. 0. 0.]] |
| FLOPS                | 25 MOPS                                                          |
| Number of parameters | 78,666                                                           |
| Source framework     | Tensorflow/Keras                                                 |
| Target platform      | MCUs, MPUs                                                       |

## Version and changelog

Initial release of quantized int8 model.

## Tested configurations

The quantized int8 models have been tested on i.MX 8M Plus using benchmark-model (see [i.MX Machine Learning User Guide](https://www.nxp.com/docs/en/user-guide/IMX-MACHINE-LEARNING-UG.pdf)).

## Training and evaluation

The model has been trained and evaluated on the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.
Training scripts are located in the model location folder on [Mlcommons Github](https://github.com/mlcommons/tiny/tree/master/benchmark/training/image_classification).

## Conversion / Quantization

The model is located in the source directory which contains original keras model (.h5), tflite float model and quantized int8 model. Quantization script can be found on the source code [GitHub page](https://github.com/mlcommons/tiny/blob/master/benchmark/training/image_classification/model_converter.py).

## Use case and limitations

Goal of the model is to classify input image into 10 classes. Classes are listed in the [Labels](#labels) section.

## Performance

Here are performance figures evaluated on i.MX 8MP using BSP LF6.1.22_1.0.0 and performance results on MCX N947 evaluated using MCUXpresso SDK, with SDK version 2.13.0 MCXN10 PRC, Toolchain MCUXpresso IDE 11.7.1 and LibC NewlibNano (nohost)::

 Model | Average latency | Platform     | Accelerator     | Command                                                                                                                                                                                     
-------|-----------------|--------------|-----------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
 Int8  | 3.83 ms         | i.MX 8M Plus | CPU (1 thread)  | /usr/bin/tensorflow-lite-2.10.0/examples/benchmark_model --graph=/usr/bin/tensorflow-lite-2.10.0/examples/pretrainedResnet_quant.tflite                                                     |
 Int8  | 1.43 ms         | i.MX 8M Plus | CPU (4 threads) | /usr/bin/tensorflow-lite-2.10.0/examples/benchmark_model --graph=/usr/bin/tensorflow-lite-2.10.0/examples/pretrainedResnet_quant.tflite --num_threads=4                                     |
 Int8  | 0.22 ms         | i.MX 8M Plus | NPU             | /usr/bin/tensorflow-lite-2.10.0/examples/benchmark_model --graph=/usr/bin/tensorflow-lite-2.10.0/examples/pretrainedResnet_quant.tflite --external_delegate_path=/usr/lib/libvx_delegate.so |
 Int8  | 1.65 ms         | i.MX 93      | CPU (1 thread)  | /usr/bin/tensorflow-lite-2.10.0/examples/benchmark_model --graph=pretrainedResnet_quant.tflite                                                                                              |
 Int8  | 1.65 ms         | i.MX 93      | CPU (2 threads) | /usr/bin/tensorflow-lite-2.10.0/examples/benchmark_model --graph=pretrainedResnet_quant.tflite --num_threads=2                                                                              |
 Int8  | 0.33ms          | i.MX 93      | NPU             | /usr/bin/tensorflow-lite-2.10.0/examples/benchmark_model --graph=pretrainedResnet_quant.tflite --external_delegate_path=/usr/lib/libethosu_delegate.so                                      |
 Int8  | 266.67 ms       | MCX N947     | CPU             | MCUXpresso SDK                                                                                                                                                                              |
 Int8  | 6.31 ms         | MCX N947     | NPU             | MCUXpresso SDK                                                                                                                                                                              

## Download and run

To download original keras model, tflite model and quantized tflite model, run `bash recipe.sh`.

The TFLite model file for i.MX 8M Plus and MCX N947 is `pretrainedResnet_quant.tflite`. Converted model for i.MX 93 is placed into `model_imx93` directory.

An example of how to use the model is in `utils/exmaple.py`. The output labels are listed in `labels.py`.

### How to run test inference:

```
 python utils/example.py --image=example_input.jpg
```

### Labels

- 'airplane',
- 'automobile',
- 'bird',
- 'cat',
- 'deer',
- 'dog',
- 'frog',
- 'horse',
- 'ship',
- 'truck'

## Origin

Model implementation: https://github.com/mlcommons/tiny/tree/master/benchmark/training/image_classification

[1]  [Learning Multiple Layers of Features from Tiny Images](http://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf),
Alex Krizhevsky, 2009.

[2] Banbury, Colby, et al. "Mlperf tiny benchmark." arXiv preprint arXiv:2106.07597 (2021).
