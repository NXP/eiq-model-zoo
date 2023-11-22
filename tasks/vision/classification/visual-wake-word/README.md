# Visual Wake Words

## Introduction

Visual Wake Words model is binary classification model. It represents a common microcontroller vision use-case of
identifying whether a person is present in the image or not.

It is based
on [MobileNet](https://github.com/SiliconLabs/platform_ml_models/blob/master/eembc/Person_detection/mobilenet_v1_eembc.py)
model architecture. We use model trained by [MLCommons](https://mlcommons.org/en/) available on
their [GitHub page](https://github.com/mlcommons/tiny/tree/master/benchmark/training/visual_wake_words).

## Model Information

| Information          | Value                                                               |
|----------------------|---------------------------------------------------------------------|
| Input shape          | RGB Color image (96, 96, 3)                                         |
| Input example        | <img src="example_input.jpg"> (Image from vww_coco2014 dataset [2]) |
| Output shape         | Vector of probabilities shape (1, 2). [[non-person, person]]        |
| Output example       | Output tensor: [[0.30859375 0.69140625]]                            |
| FLOPS                | 15 MOPS                                                             |
| Number of parameters | 221,794                                                             |
| Source framework     | Tensorflow/Keras                                                    |
| Target platform      | MCU, MPU                                                            |

## Version and changelog

Initial release of quantized int8 model.

## Tested configurations

The quantized int8 models have been tested on i.MX 8M Plus and i.MX 93 using benchmark-model (
see [i.MX Machine Learning User Guide](https://www.nxp.com/docs/en/user-guide/IMX-MACHINE-LEARNING-UG.pdf)).

## Training and evaluation

The model has been trained and evaluated on the Coco2014 dataset, which has been modified to Visual Wake Word
purposes. [Source code](https://github.com/SiliconLabs/platform_ml_models/blob/master/eembc/Person_detection/buildPersonDetectionDatabase.py)
of the dataset preparation.

## Conversion / Quantization

The model is located in a directory which contains original keras model (.h5), .tflite float model and quantized int8
model. Quantization script is located in the model source code page on
the [GitHub](https://github.com/mlcommons/tiny/blob/master/benchmark/training/visual_wake_words/convert_vww.py).

## Use case and limitations

Goal of the model is to classify whether an image contains person or not. Persons size has to be at least 2.5 % of the
overall image size.

Model is designed for tiny systems (microcontrollers).

## Performance

Here are performance figures evaluated on i.MX 8MP using BSP LF6.1.22_1.0.0 and performance results on MCX N947 evaluated using MCUXpresso SDK:

 Model | Average latency | Platform     | Accelerator     | Command                                                                                                                                                                         |
-------|-----------------|--------------|-----------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
 Int8  | 3.716 ms        | i.MX 8M Plus | CPU (1 thread)  | /usr/bin/tensorflow-lite-2.10.0/examples/benchmark_model --graph=/usr/bin/tensorflow-lite-2.10.0/examples/vww_96_int8.tflite                                                    |
 Int8  | 1.443 ms        | i.MX 8M Plus | CPU (4 threads) | /usr/bin/tensorflow-lite-2.10.0/examples/benchmark_model --graph=/usr/bin/tensorflow-lite-2.10.0/examples/vww_96_int8.tflite --num_threads=4                                    |
 Int8  | 0.483 ms        | i.MX 8M Plus | NPU             | /usr/bin/tensorflow-lite-2.10.0/examples/benchmark_model --graph=/usr/bin/tensorflow-lite-2.10.0/examples/vww_96_int8.tflite --external_delegate_path=/usr/lib/libvx_delegate.so |
 Int8  | 1.97 ms         | i.MX 93      | CPU (1 thread)  | /usr/bin/tensorflow-lite-2.10.0/examples/benchmark_model --graph=vww_96_int8.tflite                                                                                             |
 Int8  | 1.38 ms         | i.MX 93      | CPU (2 threads) | /usr/bin/tensorflow-lite-2.10.0/examples/benchmark_model --graph=vww_96_int8.tflite --num_threads=2                                                                             |
 Int8  | 0.214 ms        | i.MX 93      | NPU             | /usr/bin/tensorflow-lite-2.10.0/examples/benchmark_model --graph=vww_96_int8.tflite --external_delegate_path=/usr/lib/libethosu_delegate.so                                     |
 Int8  | 194.313 ms      | MCX N947     | CPU             | MCUXpresso SDK                                                                                                                                                                  |
 Int8  | 6.384 ms        | MCX N947     | NPU             | MCUXpresso SDK                                                                                                                                                                                |

## Download and run

To download this model run `bash recipe.sh`. This will download original keras model (.h5), float tflite model and
quantized integer model. File for the i.MX 93 will be prepared and placed into the `model_imx93` directory.

The TFLite model file for i.MX 8M Plus is `vww_96_int8.tflite`. An example of how to use the model is
in `utils/example.py`.

### How to run test inference:

```
python utils/example.py --image=example_input.jpg
```

## Origin

Model implementation: https://github.com/mlcommons/tiny/tree/master/benchmark/training/visual_wake_words

Dataset preparation: https://github.com/SiliconLabs/platform_ml_models/tree/master/eembc/Person_detection