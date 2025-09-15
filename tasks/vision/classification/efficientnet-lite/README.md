# EfficientNet-lite

## Introduction

EfficientNet[1] is a classification model designed to efficiently work on mobile devices and embedded systems by scaling the depth, width and resolution of the network with a simple coefficient.
A whole family of networks, from EfficientNet-B0 to B7 was created using Neural Architecture Search. 

EfficientNet-lite is a tweaked version of the original EfficientNet for edge devices, with the following changes:

- Remove squeeze-and-excite (SE): SE are not well supported for some mobile accelerators.
- Replace all swish with RELU6: for easier post-quantization.
- Fix the stem and head while scaling models up: for keeping models small and fast.

The model was trained on the Imagenet dataset [2]. We propose the smallest EfficientNet-lite (B0) by default. Variants 1 to 4 are available by editing `recipe.sh`.

Model implementation is from [Tensorflow GitHub page](https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/lite/README.md).

## Model Information

 Information          | Value
----------------------|------------------------------------------------------------------------------
 Input shape          | Color image (224, 224, 3)
 Input example        | <img src="example_input.jpg"> <br> ([Picture by Shizhao under CC BY-SA 3.0 license](https://commons.wikimedia.org/wiki/File:Giant_Panda_in_Beijing_Zoo_1.JPG))
 Output shape         | Vector of probabilities shape (1, 1000). Labels can be found in `labels_map.txt`.
 Output example       | Output tensor: [[-128, -123, 102, -128, ..., -128, -128, -128]]
 FLOPS                | 814 MOPS
 Number of parameters | 4.7M
 Source framework     | Tensorflow
 Target platform      | MCU, MPU

## Version and changelog

Initial release of quantized int8 model.

## Tested configurations

The quantized int8 models have been tested on i.MX 8M Plus and i.MX 93 using benchmark-model (
see [i.MX Machine Learning User Guide](https://www.nxp.com/docs/en/user-guide/IMX-MACHINE-LEARNING-UG.pdf)).

## Training and evaluation

The model has been trained and evaluated on the Imagenet dataset. It achieved a score of 75.1% Top-1 accuracy on the
test set according to [Tensorflow GitHub page](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/README.md).

The original training procedure is detailed [here](https://cloud.google.com/tpu/docs/tutorials/efficientnet).

## Conversion/Quantization

The model is downloaded directly as an int8 quantized model. The accuracy of the quantized model is 74.4%.


## Use case and limitations

This model can be used for classification applications. For better accuracy, consider using the larger variants. EfficientNet-lite4 offers 80.2% accuracy.

## Download and run

### How to run

To create the TFLite model fully quantized in int8 with int8 input and int8 output, follow the top-level README instructions to install Docker and build the Docker image, then run the following command: 

    docker run --rm -v "$PWD:/workspace" nxp-model-zoo recipe.sh

The TFLite model file for i.MX 8M Plus and for i.MX 93 is `efficientnet-lite0-int8.tflite`. 

**Note:** BSP >= LF6.1.36_2.1.0 supports Ethos-U Delegate on the i.MX93, which implements vela compilation online. If using an older BSP version, please compile the quantized TFLite model with Vela compiler before being used. Download Vela from [nxp-imx GitHub](https://github.com/nxp-imx/ethos-u-vela) from a branch, that corresponds with BSP version used.

An example of how to use the model is in `./utils/example.py`. The output labels are listed in `./utils/labels.py`.

### Run test inference

````bash
python ./example.py -i example_input.jpg -m efficientnet-lite0-int8.tflite -q
````

## Origin

Model implementation: https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/lite/README.md

[1] Tan, Mingxing, and Quoc Le. "Efficientnet: Rethinking model scaling for convolutional neural networks." International conference on machine learning. PMLR, 2019.

[2] Deng, Jia, et al. "Imagenet: A large-scale hierarchical image database." 2009 IEEE conference on computer vision and
pattern recognition. Ieee, 2009.
