# FaceNet512

## Introduction

[FaceNet](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Schroff_FaceNet_A_Unified_2015_CVPR_paper.html) is a neural network for face re-identification based on the Inception architecture.
It regresses a face feature vector that can be matched with a database of known face vectors. In this version, the face vector has a dimension of 512. We use the weights from the implementation found in [DeepFace](https://github.com/serengil/deepface/).


NB: This is not a face detection model. The faces have to be detected in the image and cropped before feeding them to FaceNet.

## Model Information

Information   | Value
---           | ---
Input shape   | RGB Face crop (160, 160, 3)
Input example | <img src="face.jpg"> (Image generated on thispersondoesnotexist.com)
Output shape  | Face feature vector of size (512)
Output example | N/A
FLOPS | 2.84 GOPS
Number of parameters | 23,497,424
Source framework | Tensorflow/Keras
Target platform | MPUs

## Version and changelog

Initial release of float32 and quantized int8 model.

## Tested configurations

The float32 and int8 models have been tested on i.MX 8MP and i.MX 93 using benchmark-model (see [i.MX Machine Learning User Guide](https://www.nxp.com/docs/en/user-guide/IMX-MACHINE-LEARNING-UG.pdf)).

## Training and evaluation

The model has been trained and evaluated on the LFW dataset. It achieved a score of 99.65% according to [the source of the model](https://github.com/serengil/deepface/).

We re-evaluated the model on [LFW aligned with deep funneling](http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz) on the [pairsDevTest](http://vis-www.cs.umass.edu/lfw/pairsDevTest.txt) split:

Model | Accuracy
---|---
FaceNet512 Keras float32 | 97.5%
FaceNet512 TensorFlow Lite int8 | 97.2%

The TensorFlow Lite model file for i.MX 8M Plus is `emotion_uint8_float32.tflite`. The file for i.MX 93 is output in the `model_imx93` directory.

The evaluation script is `evaluate.py`.

## Conversion/Quantization

The original model is directly converted from Keras to TensorFlow Lite.

The conversion script performs this conversion and outputs the float32 model and int8 quantized model. 
100 random images from the training dataset are used as calibration for the quantization.

## Use case and limitations

This model can be used for the following use cases:

- Matching a face to a database of known face vectors.

- Telling whether two persons are the same or not. 

- Tracking a person by applying the model on each frame of a video.

## Performance

Here are performance figures evaluated on i.MX 8MP:

Model   | Average latency | Platform | Accelerator | Command
---     | ---             | ---      | ---         | ---
Int8    | 322ms           | i.MX 8MP |     CPU     | /usr/bin/tensorflow-lite-2.10.0/examples/benchmark_model --graph=facenet512_uint8.tflite
Int8    | 10.0ms          | i.MX 8MP |     NPU     | /usr/bin/tensorflow-lite-2.10.0/examples/benchmark_model --graph=facenet512_uint8.tflite --external_delegate_path=/usr/lib/libvx_delegate.so
Int8    | 120.6ms         | i.MX 93  |     CPU     | /usr/bin/tensorflow-lite-2.10.0/examples/benchmark_model --graph=facenet512_uint8.tflite
Int8    | 10.3ms          | i.MX 93  |     NPU     | /usr/bin/tensorflow-lite-2.10.0/examples/benchmark_model --graph=facenet512_uint8_vela.tflite --external_delegate_path=/usr/lib/libethosu_delegate.so

## Download and run

To create the TensorFlow Lite model fully quantized in int8 with int8 input and float32 output, run:

    bash recipe.sh

An example of how to use the model is in `example.py`.

## Origin

Model implementation: https://github.com/serengil/deepface/

Original article: [Schroff, Florian, Dmitry Kalenichenko, and James Philbin. "Facenet: A unified embedding for face recognition and clustering." Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Schroff_FaceNet_A_Unified_2015_CVPR_paper.html)