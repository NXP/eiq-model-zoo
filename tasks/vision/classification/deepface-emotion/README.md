# Deepface-emotion

## Introduction

[Deepface-emotion](https://github.com/serengil/deepface) is a simple Convolutional Neural Network (CNN) for classification of face emotions.
We use the weights from the implementation found in [DeepFace](https://github.com/serengil/deepface/).

NB: This is not a face detection model. The faces have to be detected in the image and cropped before feeding them to Deepface-emotion.

## Model Information

Information   | Value
---           | ---
Input shape   | Grayscale face crop (48, 48, 1)
Input example | <img src="test.jpg"> (Image from FER2013 dataset [1])
Output shape  | Vector of probabilities shape (1, 7). Order of labels: angry, disgust, fear, happy, sad, surprise, neutral]
Output example | Output tensor: [[0., 0.,  0., 0.99609375, 0., 0., 0. ]] Recognized emotion: happy
FLOPS | 58.5 MOPS
Number of parameters | 23,497,424
Source framework | Tensorflow/Keras
Target platform | MPUs, MCUs

## Version and changelog

Initial release of float32 and quantized int8 model.

## Tested configurations

The float32 and int8 models have been tested on i.MX 8MP and i.MX 93 using benchmark-model (see [i.MX Machine Learning User Guide](https://www.nxp.com/docs/en/user-guide/IMX-MACHINE-LEARNING-UG.pdf)).

## Training and evaluation

The model has been trained and evaluated on the FER2013 dataset [1]. It achieved a score of 92.1% on the training set and 57.4% on the test set according to [the source of the model](https://github.com/serengil/deepface/).

The original training procedure is detailed [here](https://sefiks.com/2018/01/01/facial-expression-recognition-with-keras/).

We re-evaluated the floating point and int8 quantized model on the PrivateTest split:

Model | Accuracy
---|---
FaceNet512 Keras float32 | 56.5%
FaceNet512 TensorFlow Lite int8   | 56.8%

### Confusion matrix

<img src="confusion_matrix.jpg" width=500px>

The evaluation script used to create the confusion matrix is `evaluate.py`.

## Conversion/Quantization

The original model is directly converted from Keras to TensorFlow Lite.

The conversion script performs this conversion and outputs the float32 model and int8 quantized model.
100 random images from the training dataset are used as calibration for the quantization.

## Use case and limitations

This model can be used for simple emotion recognition applications. Since its accuracy on the test set is relatively low, it cannot be guaranteed to generalize well to new images.

## Download and run

To create the TensorFlow Lite model fully quantized in int8 with int8 input and float32 output, follow the top-level README instructions to install Docker and build the Docker image, then run the following command:

    docker run --rm -v "$PWD:/workspace" nxp-model-zoo recipe.sh

The TensorFlow Lite model file for i.MX 8M Plus is `emotion_uint8_float32.tflite`. The file for i.MX 93 is output in the `model_imx93` directory.

An example of how to use the model is in `example.py`.

## Origin

Model implementation: https://github.com/serengil/deepface/

[1] Goodfellow, Ian J., et al. "Challenges in representation learning: A report on three machine learning contests." International conference on neural information processing. Springer, Berlin, Heidelberg, 2013.
