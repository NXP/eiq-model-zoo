#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright 2023-2024 NXP

import argparse

import librosa
import numpy as np
import tensorflow as tf


def normalize(values):
    return (values - np.mean(values)) / np.std(values)


def transform_audio_to_mfcc(audio_file, n_mfcc=13, n_fft=512, hop_length=160):
    audio_data, sample_rate = librosa.load(audio_file, sr=16000)
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

    # add derivatives and normalize
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    mfcc = np.concatenate((normalize(mfcc), normalize(mfcc_delta), normalize(mfcc_delta2)), axis=0)
    seq_length = mfcc.shape[1] // 2

    mfcc_out = mfcc.T.astype(np.float32)
    mfcc_out = np.expand_dims(mfcc_out, 0)

    return mfcc_out, seq_length


def ctc_decoder(seq_length, y_predict):
    if len(y_predict.shape) == 4:
        y_predict = tf.squeeze(y_predict, axis=1)
    y_predict = tf.transpose(y_predict, (1, 0, 2))
    decoded, log_probabilities = tf.nn.ctc_greedy_decoder(
        y_predict, tf.cast([seq_length], tf.int32), merge_repeated=True
    )
    return tf.sparse.to_dense(decoded[0]).numpy()


def inference(model_path, data_path):
    alphabet = "abcdefghijklmnopqrstuvwxyz' @"
    index_dict = {ind: c for (ind, c) in enumerate(alphabet)}

    input_window_length = 296

    data, data_length = transform_audio_to_mfcc(data_path)

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_chunk = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_dtype = input_chunk["dtype"]
    output_dtype = output_details["dtype"]

    # Check if the input/output type is quantized,
    # set scale and zero-point accordingly
    if input_dtype != tf.float32:
        input_scale, input_zero_point = input_chunk["quantization"]
    else:
        input_scale, input_zero_point = 1, 0

    if output_dtype != tf.float32:
        output_scale, output_zero_point = output_details["quantization"]
    else:
        output_scale, output_zero_point = 1, 0

    data = data / input_scale + input_zero_point
    # Round the data up if dtype is int8, uint8 or int16
    if input_dtype is not np.float32:
        data = np.round(data)

    while data.shape[1] < input_window_length:
        data = np.append(data, data[:, -2:-1, :], axis=1)
    # Zero-pad any odd-length inputs
    if data.shape[1] % 2 == 1:
        # log('Input length is odd, zero-padding to even (first layer has stride 2)')
        data = np.concatenate([data, np.zeros((1, 1, data.shape[2]), dtype=input_dtype)], axis=1)

    context = 24 + 2 * (7 * 3 + 16)  # = 98 - theoretical max receptive field on each side
    size = input_chunk['shape'][1]  # 296
    inner = size - 2 * context  # 100
    data_end = data.shape[1]

    # Initialize variables for the sliding window loop
    data_pos = 0
    outputs = []

    while data_pos < data_end:
        if data_pos == 0:
            # Align inputs from the first window to the start of the data and include the intial context in the output
            start = data_pos
            end = start + size
            y_start = 0
            y_end = y_start + (size - context) // 2
            data_pos = end - context
        elif data_pos + inner + context >= data_end:
            # Shift left to align final window to the end of the data and include the final context in the output
            shift = (data_pos + inner + context) - data_end
            start = data_pos - context - shift
            end = start + size
            assert start >= 0
            y_start = (shift + context) // 2  # Will be even because we assert it above
            y_end = size // 2
            data_pos = data_end
        else:
            # Capture only the inner region from mid-input inferences, excluding output from both context regions
            start = data_pos - context
            end = start + size
            y_start = context // 2
            y_end = y_start + inner // 2
            data_pos = end - context

        interpreter.set_tensor(input_chunk["index"], tf.cast(data[:, start:end, :], input_dtype))
        interpreter.invoke()
        cur_output_data = interpreter.get_tensor(output_details["index"])[:, :, y_start:y_end, :]
        cur_output_data = output_scale * (
                cur_output_data.astype(np.float32) - output_zero_point
        )
        outputs.append(cur_output_data)
    complete = np.concatenate(outputs, axis=2)
    output = ctc_decoder(data_length, complete)
    decoded_output = [index_dict[value] for value in output[0]]
    return decoded_output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default='wav2letter_int8.tflite', type=str)
    parser.add_argument("--data", default='example_input.flac', type=str)

    args = parser.parse_args()

    output = inference(args.model_path, args.data)
    print(f'Transcribed File: {"".join(output)}')
