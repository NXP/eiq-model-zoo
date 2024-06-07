#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright 2023-2024 NXP

import librosa
import functools
import numpy as np
import tensorflow as tf

import model_settings as models
import kws_util

CLASSES = [
    "Down",
    "Go",
    "Left",
    "No",
    "Off",
    "On",
    "Right",
    "Stop",
    "Up",
    "Yes",
    "Silence",
    "Unknown",
]


def prepare_processing_graph(input_data, model_settings):
    """Builds a TensorFlow graph to apply the input distortions.
    Creates a graph that loads a WAVE file, decodes it, scales the volume,
    shifts it in time, adds in background noise, calculates a spectrogram, and
    then builds an MFCC fingerprint from that.
    This must be called with an active TensorFlow session running, and it
    creates multiple placeholder inputs, and one output:
      - wav_filename_placeholder_: Filename of the WAV to load.
      - foreground_volume_placeholder_: How loud the main clip should be.
      - time_shift_padding_placeholder_: Where to pad the clip.
      - time_shift_offset_placeholder_: How much to move the clip in time.
      - background_data_placeholder_: PCM sample data for background noise.
      - background_volume_placeholder_: Loudness of mixed-in background.
      - mfcc_: Output 2D fingerprint of processed audio.
    Args:
      model_settings: Information about the current model being trained.
    """
    desired_samples = model_settings["desired_samples"]
    # background_frequency = model_settings['background_frequency']
    # background_volume_range_ = model_settings['background_volume_range_']

    wav_decoder = tf.cast(input_data, tf.float32)
    if model_settings["feature_type"] != "td_samples":
        wav_decoder = wav_decoder / tf.reduce_max(wav_decoder)
    else:
        wav_decoder = wav_decoder / tf.constant(2**15, dtype=tf.float32)
    # Previously, decode_wav was used with desired_samples as the length of array. The
    # default option of this function was to pad zeros if the desired samples are not

    wav_decoder = tf.pad(
        wav_decoder, [[0, desired_samples - tf.shape(wav_decoder)[-1]]]
    )
    # Allow the audio sample's volume to be adjusted.
    foreground_volume_placeholder_ = tf.constant(1, dtype=tf.float32)

    scaled_foreground = tf.multiply(wav_decoder, foreground_volume_placeholder_)
    # Shift the sample's start position, and pad any gaps with zeros.
    time_shift_padding_placeholder_ = tf.constant([[2, 2]], tf.int32)
    time_shift_offset_placeholder_ = tf.constant([2], tf.int32)
    # scaled_foreground.shape
    padded_foreground = tf.pad(
        scaled_foreground, time_shift_padding_placeholder_, mode="CONSTANT"
    )
    sliced_foreground = tf.slice(
        padded_foreground, time_shift_offset_placeholder_, [desired_samples]
    )

    if model_settings["feature_type"] == "mfcc":
        stfts = tf.signal.stft(
            sliced_foreground,
            frame_length=model_settings["window_size_samples"],
            frame_step=model_settings["window_stride_samples"],
            fft_length=None,
            window_fn=tf.signal.hann_window,
        )
        spectrograms = tf.abs(stfts)
        num_spectrogram_bins = stfts.shape[-1]
        # default values used by contrib_audio.mfcc as shown here
        # https://kite.com/python/docs/tensorflow.contrib.slim.rev_block_lib.contrib_framework_ops.audio_ops.mfcc
        lower_edge_hertz, upper_edge_hertz, num_mel_bins = 20.0, 4000.0, 40
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins,
            num_spectrogram_bins,
            model_settings["sample_rate"],
            lower_edge_hertz,
            upper_edge_hertz,
        )
        mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
        mel_spectrograms.set_shape(
            spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:])
        )
        # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
        log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
        # Compute MFCCs from log_mel_spectrograms and take the first 13.
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[
            ..., : model_settings["dct_coefficient_count"]
        ]
        mfccs = tf.reshape(
            mfccs,
            [
                model_settings["spectrogram_length"],
                model_settings["dct_coefficient_count"],
                1,
            ],
        )
        input_data = mfccs

    elif model_settings["feature_type"] == "lfbe":
        # apply preemphasis
        preemphasis_coef = 1 - 2**-5
        power_offset = 52
        num_mel_bins = model_settings["dct_coefficient_count"]
        paddings = tf.constant([[0, 0], [1, 0]])
        # for some reason, tf.pad only works with the extra batch dimension, but then we remove it after pad
        sliced_foreground = tf.expand_dims(sliced_foreground, 0)
        sliced_foreground = tf.pad(
            tensor=sliced_foreground, paddings=paddings, mode="CONSTANT"
        )
        sliced_foreground = (
            sliced_foreground[:, 1:] - preemphasis_coef * sliced_foreground[:, :-1]
        )
        sliced_foreground = tf.squeeze(sliced_foreground)
        # compute fft
        stfts = tf.signal.stft(
            sliced_foreground,
            frame_length=model_settings["window_size_samples"],
            frame_step=model_settings["window_stride_samples"],
            fft_length=None,
            window_fn=functools.partial(tf.signal.hamming_window, periodic=False),
            pad_end=False,
            name="STFT",
        )

        # compute magnitude spectrum [batch_size, num_frames, NFFT]
        magspec = tf.abs(stfts)
        num_spectrogram_bins = magspec.shape[-1]

        # compute power spectrum [num_frames, NFFT]
        powspec = (1 / model_settings["window_size_samples"]) * tf.square(magspec)
        powspec_max = tf.reduce_max(input_tensor=powspec)
        powspec = tf.clip_by_value(
            powspec, 1e-30, powspec_max
        )  # prevent -infinity on log

        def log10(x):
            # Compute log base 10 on the tensorflow graph.
            # x is a tensor.  returns log10(x) as a tensor
            numerator = tf.math.log(x)
            denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
            return numerator / denominator

        # Warp the linear-scale, magnitude spectrograms into the mel-scale.
        lower_edge_hertz, upper_edge_hertz = 0.0, model_settings["sample_rate"] / 2.0
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=num_mel_bins,
            num_spectrogram_bins=num_spectrogram_bins,
            sample_rate=model_settings["sample_rate"],
            lower_edge_hertz=lower_edge_hertz,
            upper_edge_hertz=upper_edge_hertz,
        )

        mel_spectrograms = tf.tensordot(powspec, linear_to_mel_weight_matrix, 1)
        mel_spectrograms.set_shape(
            magspec.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:])
        )

        log_mel_spec = 10 * log10(mel_spectrograms)
        log_mel_spec = tf.expand_dims(log_mel_spec, -1, name="mel_spec")

        log_mel_spec = (log_mel_spec + power_offset - 32 + 32.0) / 64.0
        log_mel_spec = tf.clip_by_value(log_mel_spec, 0, 1)

        input_data = log_mel_spec

    elif model_settings["feature_type"] == "td_samples":
        ## sliced_foreground should have the right data.  Make sure it's the right format (int16)
        # and just return it.
        paddings = [[0, 16000 - tf.shape(sliced_foreground)[0]]]
        wav_padded = tf.pad(sliced_foreground, paddings)
        wav_padded = tf.expand_dims(wav_padded, -1)
        wav_padded = tf.expand_dims(wav_padded, -1)
        input_data = wav_padded

    input_data = np.expand_dims(input_data, axis=0)
    return input_data


def quantize(input, scale, zp):
    return (input / scale) + zp


def run_inference(model, data):
    interpreter = tf.lite.Interpreter(model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("Loaded model")

    scale, zero_point = input_details[0]["quantization"]
    data = quantize(data, scale, zero_point)
    input_data = data.astype(dtype=np.int8)

    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]["index"])

    out_quant = output_details[0]["quantization"]
    detection_classes = out.astype(np.float32)
    detection_classes = out_quant[0] * (detection_classes - out_quant[1])
    print("Recognized word:", CLASSES[detection_classes.argmax()])


def prepare_data(Flags):
    soundfile, sr = librosa.load(Flags.file)
    model_settings = models.prepare_model_settings(len(CLASSES), sr, Flags)

    data = prepare_processing_graph(soundfile, model_settings)

    run_inference(Flags.tfl_file_name, data)


if __name__ == "__main__":
    Flags, unparsed = kws_util.parse_command()
    prepare_data(Flags)
