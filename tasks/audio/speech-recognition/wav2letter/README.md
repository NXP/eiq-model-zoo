# Wav2Letter

## Introduction
Wav2Letter is  a simple end-to-end model for speech recognition, combining a convolutional network based acoustic model and a graph decoding. It is trained
to output letters, with transcribed speech, without the need for force alignment of phonemes. [1]


## Model Information

Information   | Value
---           | ---
Input shape   | (1, 296, 39)
Input example | "example_input.flac". Source: Librispeech dataset [2] License: CC BY 4.0
Output shape  | (1, 1, 148, 29)
Output example | Output tensor: [[0. 0. 0. ... 0. 0.01953125]
FLOPS |  6 982 MOPS
Number of parameters | -
Source framework | Tensorflow Lite
Target platform | MPU, MCU

## Version and changelog

Initial release of quantized int8 model.

## Tested configurations

The quantized int8 models have been tested on i.MX 8M Plus and i.MX 93 using benchmark-model (see [i.MX Machine Learning User Guide](https://www.nxp.com/docs/en/user-guide/IMX-MACHINE-LEARNING-UG.pdf)).

## Training and evaluation

The model has been trained and evaluated on the Librispeech dataset [2]. It achieved a score of 7.2% WER (word error rate) on the test set according to the paper [1].

The original training procedure is detailed [here](https://github.com/flashlight/wav2letter).

## Conversion/Quantization

The model is downloaded in the quantized int8 tflite format. See [the source of the model](https://github.com/ARM-software/ML-zoo/tree/master/models/speech_recognition/wav2letter) for information on the quantization procedure that was used.

## Use case and limitations

This model can be used for speech recognition applications.

## Download and run

To create the TFLite model fully quantized in int8 with int8 input and int8 output, follow the top-level README instructions to install Docker and build the Docker image, then run the following command: 

    docker run --rm -v "$PWD:/workspace" nxp-model-zoo recipe.sh run `bash recipe.sh`.

The TFLite model file for i.MX 8M Plus and i.MX93 is `.tflite`. 

**Note:** BSP >= LF6.1.36_2.1.0 supports Ethos-U Delegate on the i.MX93, which implements vela compilation online. If using an older BSP version, please compile the quantized TFLite model with Vela compiler before being used. Download Vela from [nxp-imx GitHub](https://github.com/nxp-imx/ethos-u-vela) from a branch, that corresponds with BSP version used.


An example of how to use the model is in `example.py`.

## Origin

[1] Collobert, Ronan, Christian Puhrsch, and Gabriel Synnaeve. "Wav2letter: an end-to-end convnet-based speech recognition system." arXiv preprint arXiv:1609.03193 (2016).

[2] Panayotov, Vassil, et al. "Librispeech: an asr corpus based on public domain audio books." 2015 IEEE international conference on acoustics, speech and signal processing (ICASSP). IEEE, 2015.
