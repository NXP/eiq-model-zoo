#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright 2023-2024 NXP

import argparse
import sys

import librosa
import numpy
import tensorflow as tf


def file_load(wav_name):
    """
    load .wav file.
    wav_name : str
        target .wav file
    sampling_rate : int
        audio file sampling_rate
    return : numpy.array( float )
    """
    try:
        return librosa.load(wav_name, sr=None, mono=False)
    except:
        print("file_broken or not exists!! : {}".format(wav_name))


def quantize(input, scale, zp):
    return (input / scale) + zp


def file_to_vector_array(
    file_name,
    n_mels=128,
    frames=5,
    n_fft=1024,
    hop_length=512,
    power=2.0,
    method="librosa",
):
    """
    convert file_name to a vector array.

    file_name : str
        target .wav file

    return : numpy.array( numpy.array( float ) )
        vector array
        * dataset.shape = (dataset_size, feature_vector_length)
    """
    # 01 calculate the number of dimensions
    dims = n_mels * frames

    # 02 generate melspectrogram
    y, sr = file_load(file_name)
    if method == "librosa":
        # 02a generate melspectrogram using librosa
        mel_spectrogram = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=power
        )

        # 03 convert melspectrogram to log mel energy
        log_mel_spectrogram = (
            20.0 / power * numpy.log10(mel_spectrogram + sys.float_info.epsilon)
        )

    else:
        print("spectrogram method not supported: {}".format(method))
        return numpy.empty((0, dims))

    # 3b take central part only
    log_mel_spectrogram = log_mel_spectrogram[:, 50:250]

    # 04 calculate total vector size
    vector_array_size = len(log_mel_spectrogram[0, :]) - frames + 1

    # 05 skip too short clips
    if vector_array_size < 1:
        return numpy.empty((0, dims))

    # 06 generate feature vectors by concatenating multiframes
    vector_array = numpy.zeros((vector_array_size, dims))
    for t in range(frames):
        vector_array[:, n_mels * t : n_mels * (t + 1)] = log_mel_spectrogram[
            :, t : t + vector_array_size
        ].T

    return vector_array


def predict(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    output_data = numpy.empty_like(input_data)

    for i in range(input_data.shape[0]):
        interpreter.set_tensor(input_details[0]["index"], input_data[i : i + 1, :])
        interpreter.invoke()

        output_data[i : i + 1, :] = interpreter.get_tensor(output_details[0]["index"])

    scale, zp = output_details[0]["quantization"]

    out = output_data.astype(numpy.float32)
    out = scale * (out - zp)

    return output_data, out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="ad01_int8.tflite", type=str)
    parser.add_argument("--data", default="example_input.wav", type=str)
    args = parser.parse_args()

    data_fp = file_to_vector_array(args.data)

    interpreter = tf.lite.Interpreter(model_path=args.model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]

    scale, zero_point = input_details["quantization"]

    data = quantize(data_fp, scale, zero_point)

    input_data = data.astype(dtype=numpy.int8)

    out, out_dequant = predict(interpreter, input_data)

    errors = numpy.mean(numpy.square(data_fp - out_dequant), axis=1)
    y_pred = numpy.mean(errors)

    print("Prediction ", y_pred)
