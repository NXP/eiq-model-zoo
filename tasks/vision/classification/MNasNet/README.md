# MNasNet

## Introduction

[MNasNet](https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet) is a mobile hardware friendly neural network, found by Mobile Neural Architecture Search [1]. Goal of the model is image classification, trained on the Imagenet dataset [2].

## Model Information

 Information          | Value
----------------------|--------------------------------------------------------------------------------------------
 Input shape          | Color image (224, 224, 3)
 Input example        | <img src="example_input.jpg" width=320px> <br>  ([Public domain picture](https://commons.wikimedia.org/wiki/File:A_pure_and_female_Boxer_dog_in_Iran_10.jpg))
 Output shape         | Vector of probabilities shape (1, 1000). Labels can be found in `./utils/labels.py`.
 Output example       | Output tensor: [[0., ... 0.03125, 0.44921875, 0., 0., 0., 0., 0., 0., 0.015625, 0., ... ]]
 FLOPS                | 447 MOPS
 Number of parameters | 2.9 M
 Source framework     | Tensorflow/Keras
 Target platform      | MCU, MPU

## Version and changelog

Initial release of quantized int8 model.

## Tested configurations

The quantized int8 models have been tested on i.MX 8M Plus and i.MX93 using benchmark-model (
see [i.MX Machine Learning User Guide](https://www.nxp.com/docs/en/user-guide/IMX-MACHINE-LEARNING-UG.pdf))

## Training and evaluation

The model has been trained and evaluated on the Imagenet dataset. It achieved a score of 75.2% top-1 accuracy on the test set according to [1].

The original training procedure is detailed in [the source of the model](https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet).

## Conversion/Quantization

Downloaded directory contains keras model saved in the `keras_model/mnasnet-a1-075` directory with `.pb` file. Following converted and quantized models were prepared. Script for model conversion is placed in the `utils` directory.

### Prepared models

1. mnasnet_a1_075_float32.tflite - Original model converted to TFLite format
2. mnasnet_a1_075_quant_int8.tflite - Quantized model (int8 input, int8 output) in TFLite format, for i.MX 8MP and i.MX 93

From BSP LF6.1.36_2.1.0 Ethos-U delegate, which implements vela compilation online, is supported. In case an older version of BSP is used, vela needs to be downloaded and model has to be converted beforehand.

## Use case and limitations

This model can be used for image classification applications.

As of BSP LF6.1.36_2.1.0, Ethos-U delegate does not support some operations in the model. Therefore, model cannot be accelerated on i.MX 93 NPU.

## Performance

Here are performance figures evaluated on i.MX 8MP and i.MX 93 using BSP LF6.1.36_2.1.0:

Model | Average latency | Platform     | Accelerator     | Command
------|-----------------|--------------|-----------------|---------------------------------------------------------
Int8  | 113.68 ms       | i.MX 8M Plus | CPU (1 thread)  | benchmark_model --graph=mnasnet-a1-075_quant_int8.tflite
Int8  | 43.60 ms        | i.MX 8M Plus | CPU (4 threads) | benchmark_model --graph=mnasnet-a1-075_quant_int8.tflite --num_threads=4
Int8  | 12.95 ms        | i.MX 8M Plus | NPU             | benchmark_model --graph=mnasnet-a1-075_quant_int8.tflite --external_delegate_path=libvx_delegate.so
Int8  | 66.91 ms        | i.MX 93      | CPU (1 thread)  | benchmark_model --graph=mnasnet-a1-075_quant_int8.tflite
Int8  | 45.71 ms        | i.MX 93      | CPU (2 threads) | benchmark_model --graph=mnasnet-a1-075_quant_int8.tflite --num_threads=2
Int8  | -               | i.MX 93      | NPU             | benchmark_model --graph=mnasnet-a1-075_quant_int8.tflite --external_delegate_path=libethosu_delegate.so

Refer to the [User Guide](https://www.nxp.com/docs/en/user-guide/IMX-MACHINE-LEARNING-UG.pdf) to locate benchmark model and external delegates paths.


## Download and run

### How to build models

1. Download and convert model by running `recipe.sh`.

### How to run test inference

1. Run quantized (int8) model inference with input image:

```bash
python3 utils/example.py -m mnasnet-a1-075_quant_int8.tflite -i example_input.jpg -q
```

1c. Run non-quantized (float32) model inference with input tensor:

```bash
python3 utils/example.py -m mnasnet-a1-075_float32.tflite -i example_input.jpg
```

## Origin

Model implementation: https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet

[1] Mingxing Tan, Bo Chen, Ruoming Pang, Vijay Vasudevan, Mark Sandler, Andrew Howard, Quoc V. Le. MnasNet:
Platform-Aware Neural Architecture Search for Mobile. CVPR 2019. Arxiv link: https://arxiv.org/abs/1807.11626

[2] Deng, Jia, et al. "Imagenet: A large-scale hierarchical image database." 2009 IEEE conference on computer vision and
pattern recognition. Ieee, 2009.
