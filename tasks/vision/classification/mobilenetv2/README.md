# MobileNet V2

## Introduction

MobileNet V2 is a classification model designed to efficiently work on mobile devices and embedded systems. It is built upon the original MobileNet. This model (v2) uses inverted residual blocks with bottlenecking features [1]. Model was trained on the Imagenet dataset [2].

Model implementation is placed on [Tensorflow GitHub page](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet).

## Model Information

 Information          | Value
----------------------|------------------------------------------------------------------------------
 Input shape          | Color image (224, 224, 3)
 Input example        | <img src="example_input.jpg"> <br> ([Public domain picture](https://commons.wikimedia.org/wiki/File:A_pure_and_female_Boxer_dog_in_Iran_10.jpg))
 Output shape         | Vector of probabilities shape (1, 1000). Labels can be found in `./utils/labels.py`.
 Output example       | Output tensor: [[-128, -123, 102, -128, ..., -128, -128, -128]]
 FLOPS                | 608 MOPS
 Number of parameters | 3,538,984
 Source framework     | Tensorflow/Keras
 Target platform      | MCU, MPU

## Version and changelog

Initial release of quantized int8 model.

## Tested configurations

The quantized int8 models have been tested on i.MX 8M Plus and i.MX 93 using benchmark-model (
see [i.MX Machine Learning User Guide](https://www.nxp.com/docs/en/user-guide/IMX-MACHINE-LEARNING-UG.pdf)).

## Training and evaluation

The model has been trained and evaluated on the Imagenet dataset. It achieved a score of 71.8% Top-1 accuracy on the
test set according to [Tensorflow GitHub page](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/README.md).

The original training procedure is detailed [here](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v2.py).

## Conversion/Quantization

The model is downloaded via Tensorflow library in a keras format and converted to tflite float model and quantized int8 model. Quantization script is located in the `utils` directory.


## Use case and limitations

This model can be used for classification applications. Since its accuracy is relatively low, it should not be trusted for critical scenarios.

## Performance

Here are performance figures evaluated on i.MX 8MP and i.MX 93 using BSP LF6.1.36_2.1.0:

Model | Average latency | Platform     | Accelerator     | Command
------|-----------------|--------------|-----------------|---------------------------------------------------------
Int8  | 123.31 ms       | i.MX 8M Plus | CPU (1 thread)  | benchmark_model --graph=mobilenet_v2_quant_int8.tflite
Int8  | 44.33 ms        | i.MX 8M Plus | CPU (4 threads) | benchmark_model --graph=mobilenet_v2_quant_int8.tflite --num_threads=4
Int8  | 6.96 ms         | i.MX 8M Plus | NPU             | benchmark_model --graph=mobilenet_v2_quant_int8.tflite --external_delegate_path=libvx_delegate.so
Int8  | 62.51 ms        | i.MX 93      | CPU (1 thread)  | benchmark_model --graph=mobilenet_v2_quant_int8.tflite
Int8  | 43.96 ms        | i.MX 93      | CPU (2 threads) | benchmark_model --graph=mobilenet_v2_quant_int8.tflite --num_threads=2
Int8  | 3.46 ms         | i.MX 93      | NPU             | benchmark_model --graph=mobilenet_v2_quant_int8.tflite --external_delegate_path=libethosu_delegate.so

Note: Refer to the [User Guide](https://www.nxp.com/docs/en/user-guide/IMX-MACHINE-LEARNING-UG.pdf), to find out where benchmark_model, libvx_delegate and libethosu_delegate are located.

## Download and run

### How to run

To create the TFLite model fully quantized in int8 with int8 input and int8 output, run `bash recipe.sh`.

The TFLite model file for i.MX 8M Plus and for i.MX 93 is `.tflite`. 

**Note:** BSP >= LF6.1.36_2.1.0 supports Ethos-U Delegate on the i.MX93, which implements vela compilation online. If using an older BSP version, please compile the quantized TFLite model with Vela compiler before being used. Download Vela from [nxp-imx GitHub](https://github.com/nxp-imx/ethos-u-vela) from a branch, that corresponds with BSP version used.

An example of how to use the model is in `./utils/example.py`. The output labels are listed in `./utils/labels.py`.

### Run test inference

````bash
python ./utils/example.py -i example_input.jpg -m mobilenet_v2_quant_int8.tflite -q
````

## Origin

Model implementation: https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet

[1] Mark Sandler, et al. "MobileNetV2: Inverted Residuals and Linear Bottlenecks" arXiv preprint arXiv:1801.04381 (
2019).

[2] Deng, Jia, et al. "Imagenet: A large-scale hierarchical image database." 2009 IEEE conference on computer vision and
pattern recognition. Ieee, 2009.
