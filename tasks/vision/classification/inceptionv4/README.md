# Inception v4

## Introduction

Inception v4 [1] is part of Inception family models. Model is trained for image classification task. In comparison to previous version, Inception v4 has more uniform and simplified architecture and more inception modules [1].

Model was trained on Imagenet dataset [2].

## Model Information

 Information          | Value
----------------------|------------------------------------------------------------------------------
 Input shape          | Color image (299, 299, 3)
 Input example        | <img src="example_input.jpg"> <br> ([Public domain picture](https://commons.wikimedia.org/wiki/File:A_pure_and_female_Boxer_dog_in_Iran_10.jpg))
 Output shape         | Vector of probabilities shape (1, 1000). Labels can be found in `./labels.py`.
 Output example       | Output tensor: [[0, ..., 0, 252, 1, ..., 0]]
 FLOPS                | 24.6 G OPS
 Number of parameters | 43 M
 Source framework     | Tensorflow
 Target platform      | MPU

## Version and changelog

Initial release of quantized int8 model.

## Tested configurations

The quantized int8 models have been tested on i.MX 8M Plus and i.MX 93 using benchmark-model (
see [i.MX Machine Learning User Guide](https://www.nxp.com/docs/en/user-guide/IMX-MACHINE-LEARNING-UG.pdf)).

## Training and evaluation

The model has been trained and evaluated on the Imagenet dataset. It achieved a score of 80.2% Top-1 accuracy on the
test set according to [TensorFlow GitHub documentation](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models)


## Conversion/Quantization

The model is downloaded from [Kaggle](https://www.kaggle.com/models/tensorflow/inception/tfLite/v4-quant) in the int8 quantized tflite format.

## Use case and limitations

This model can be used for classification applications. Since its accuracy is relatively low, it should not be trusted for critical scenarios.

## Download and run

### How to run

To create the TFLite model fully quantized in int8 with int8 input and int8 output, follow the top-level README instructions to install Docker and build the Docker image, then run the following command: 

    docker run --rm -v "$PWD:/workspace" nxp-model-zoo recipe.sh

The TFLite model file for i.MX 8M Plus and for i.MX 93 is `inceptionv4_quant_int8.tflite`. 

**Note:** BSP >= LF6.1.36_2.1.0 supports Ethos-U Delegate on the i.MX93, which implements vela compilation online. If using an older BSP version, please compile the quantized TFLite model with Vela compiler before being used. Download Vela from [nxp-imx GitHub](https://github.com/nxp-imx/ethos-u-vela) from a branch, that corresponds with BSP version used.

An example of how to use the model is in `./example.py`. The output labels are listed in `./labels.py`.

### Run test inference

````bash
python example.py 
````

## Origin

[1] Szegedy, Christian, et al. "Inception-v4, inception-resnet and the impact of residual connections on learning." Proceedings of the AAAI conference on artificial intelligence. Vol. 31. No. 1. 2017.

[2] Deng, Jia, et al. "Imagenet: A large-scale hierarchical image database." 2009 IEEE conference on computer vision and
pattern recognition. Ieee, 2009.
