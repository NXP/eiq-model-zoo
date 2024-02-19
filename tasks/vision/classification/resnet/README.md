# ResNet50

## Introduction

ResNet50 is a deep convolutional neural network architecture composed od 50 layers. It was originally developed to address the vanishing gradient problem in deep networks. It introduced residual connections. ResNet50 is widely used for image classification tasks.

In this model package we use ResNet50 with an improved training procedure, explained in the paper [2]. 

## Model Information

Information   | Value
---           | ---
Input shape   | Color image (224, 224, 3)
Input example | <img src="example_input.jpg"> <br> ([Public domain picture](https://commons.wikimedia.org/wiki/File:A_pure_and_female_Boxer_dog_in_Iran_10.jpg))
Output shape  | Vector of probabilities shape (1, 1000). Labels can be found in `./labels.py`.
Output example | Output tensor: [[-128, -123, 102, -128, ..., -128, -128, -128]]
FLOPS | 6 977 MOPS
Number of parameters |  25,613,800
Source framework | Tensorflow/Keras
Target platform | MCU, MPU

## Version and changelog

Initial release of quantized int8 model.

## Tested configurations

The quantized int8 models have been tested on i.MX 8M Plus and i.MX 93 using benchmark-model (see [i.MX Machine Learning User Guide](https://www.nxp.com/docs/en/user-guide/IMX-MACHINE-LEARNING-UG.pdf)).

## Training and evaluation

The model has been trained and evaluated on the Imagenet dataset [3]. It achieved a score of 80% Top-1 accuracy on the test set according to paper [2].

The improved training procedure is detailed in the paper [2].

## Conversion/Quantization

The model is accessed through keras API and converted and quantized into TensorFlow Lite int8 model.

## Use case and limitations

This model can be used for image classification applications. Since its accuracy is relatively low, it should not be trusted for critical scenarios.

## Performance

Here are performance figures evaluated on i.MX 8MP and i.MX 93  using BSP LF6.1.36_2.1.0:

Model   | Average latency | Platform     | Accelerator     | Command
---     |-----------------|--------------|-----------------| ---
Int8    | 836 ms          | i.MX 8M Plus | CPU (1 thread)  | benchmark_model --graph=resnet50_int8.tflite
Int8    | 259 ms          | i.MX 8M Plus | CPU (4 threads) | benchmark_model --graph=resnet50_int8.tflite --num_threads=4
Int8    | 25 ms           | i.MX 8M Plus | NPU             | benchmark_model --graph=resnet50_int8.tflite--external_delegate_path=libvx_delegate.so
Int8    | 303 ms          | i.MX 93      | CPU (1 thread)  | benchmark_model --graph=resnet50_int8.tflite
Int8    | 194 ms          | i.MX 93      | CPU (2 threads) | benchmark_model --graph=resnet50_int8.tflite --num_threads=2
Int8    | 28 ms           | i.MX 93      | NPU             | benchmark_model --graph=resnet50_int8.tflite --external_delegate_path=libethosu_delegate.so

Note: Refer to the [User Guide](https://www.nxp.com/docs/en/user-guide/IMX-MACHINE-LEARNING-UG.pdf), to find out where benchmark_model, libvx_delegate and libethosu_delegate are located.


## Download and run

To create the TFLite model fully quantized in int8 with int8 input and int8 output, run `bash recipe.sh`.

The TFLite model file for i.MX 8M Plus and i.MX 93 is `.tflite`.

**Note:** BSP >= LF6.1.36_2.1.0 supports Ethos-U Delegate on the i.MX93, which implements vela compilation online. If using an older BSP version, please compile the quantized TFLite model with Vela compiler before being used. Download Vela from [nxp-imx GitHub](https://github.com/nxp-imx/ethos-u-vela) from a branch, that corresponds with BSP version used.


An example of how to use the model is in `example.py`. The output labels are listed in `labels.py`.

### Run test inference

````bash
python ./utils/example.py -i example_input.jpg -m resnet50_int8.tflite -q
````

## Origin

Model implementation: 

[1] He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
[2] Wightman, Ross, Hugo Touvron, and Hervé Jégou. "Resnet strikes back: An improved training procedure in timm." arXiv preprint arXiv:2110.00476 (2021).
[3] Deng, Jia, et al. "Imagenet: A large-scale hierarchical image database." 2009 IEEE conference on computer vision and pattern recognition. Ieee, 2009.
