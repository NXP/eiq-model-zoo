#!/usr/bin/env python

# SPDX-License-Identifier: MIT
# Copyright 2023 NXP

import os
import pathlib
import zipfile

import numpy as np
import tensorflow as tf
import argparse
import shutil
from tqdm import tqdm

AUTOTUNE = tf.data.AUTOTUNE
data_dir = pathlib.Path("data/mini_speech_commands")


def prepare_dataset(data_dir):
    if not data_dir.exists():
        tf.keras.utils.get_file(
            "mini_speech_commands.zip",
            origin="https://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
            extract=True,
            cache_dir=".",
            cache_subdir="data",
        )
        # Moving wav files from command directories to unknown sub-directory (Factory Reset to reset data directory)
        # comment out below line if "unknown" directory already exists
        os.system("mkdir ./data/mini_speech_commands/unknown")
        # moves files from their specific commands directory to the "unknown" directory (replaces all files with an existing name therefore example set is smaller)
        os.system(
            "mv ./data/mini_speech_commands/down/* ./data/mini_speech_commands/unknown"
        )
        os.system("rm -d ./data/mini_speech_commands/down")
        os.system(
            "mv ./data/mini_speech_commands/go/* ./data/mini_speech_commands/unknown"
        )
        os.system("rm -d ./data/mini_speech_commands/go")
        os.system(
            "mv ./data/mini_speech_commands/left/* ./data/mini_speech_commands/unknown"
        )
        os.system("rm -d ./data/mini_speech_commands/left")
        os.system(
            "mv ./data/mini_speech_commands/right/* ./data/mini_speech_commands/unknown"
        )
        os.system("rm -d ./data/mini_speech_commands/right")
        os.system(
            "mv ./data/mini_speech_commands/stop/* ./data/mini_speech_commands/unknown"
        )
        os.system("rm -d ./data/mini_speech_commands/stop")
        os.system(
            "mv ./data/mini_speech_commands/up/* ./data/mini_speech_commands/unknown"
        )
        os.system("rm -d ./data/mini_speech_commands/up")
        os.system("rm ./data/mini_speech_commands/README.md")
        os.system("ls ./data/mini_speech_commands/unknown | wc -l")

        print("Dataset prepared in {}.".format(data_dir))
    else:
        print("Dataset already available in {}.".format(data_dir))


# ## Spectrogram
# You'll convert the waveform into a spectrogram, which shows frequency changes over time and can be represented as a 2D image.
# This can be done by applying the short-time Fourier transform (STFT) to convert the audio into the time-frequency domain.
#
# A Fourier transform ([`tf.signal.fft`](https://www.tensorflow.org/api_docs/python/tf/signal/fft)) converts a signal to
# its component frequencies, but loses all time information. The STFT ([`tf.signal.stft`](https://www.tensorflow.org/api_docs/python/tf/signal/stft))
# splits the signal into windows of time and runs a Fourier transform on each window, preserving some time information,
# and returning a 2D tensor that you can run standard convolutions on.
# STFT produces an array of complex numbers representing magnitude and phase. However, you'll only need the magnitude for
# this tutorial, which can be derived by applying `tf.abs` on the output of `tf.signal.stft`.
# Choose `frame_length` and `frame_step` parameters such that the generated spectrogram "image" is almost square. For more
# information on STFT parameters choice, you can refer to [this video](https://www.coursera.org/lecture/audio-signal-processing/stft-2-tjEQe)
# on audio signal processing.
# You also want the waveforms to have the same length, so that when you convert it to a spectrogram image, the results
# will have similar dimensions. This can be done by simply zero padding the audio clips that are shorter than one second.
def get_spectrogram(waveform):
    # Padding for files with less than 16000 samples
    zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)

    # Concatenate audio with padding so that all audio clips will be of the
    # same length
    waveform = tf.cast(waveform, tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    spectrogram = tf.signal.stft(
        equal_length, frame_length=480, frame_step=320, fft_length=512
    )
    spectrogram = tf.abs(spectrogram)
    return spectrogram


def get_spectrogram_and_label_id(audio, label, file_name):
    commands = get_commands(data_dir=data_dir)
    spectrogram = get_spectrogram(audio)
    spectrogram = tf.expand_dims(spectrogram, -1)
    label_id = tf.argmax(label == commands)
    return spectrogram, label_id, file_name


def preprocess_dataset(files):
    # AUTOTUNE = tf.data.AUTOTUNE
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    output_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
    output_ds = output_ds.map(get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)
    return output_ds


# The label for each WAV file is its parent directory.
def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)

    # Note: You'll use indexing here instead of tuple unpacking to enable this
    # to work in a TensorFlow graph.
    return parts[-2]


def get_filename(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-1]


# ## Reading audio files and their labels
# The audio file will initially be read as a binary file, which you'll want to convert into a numerical tensor.
# To load an audio file, you will use [`tf.audio.decode_wav`](https://www.tensorflow.org/api_docs/python/tf/audio/decode_wav), which returns the WAV-encoded audio as a Tensor and the sample rate.
# A WAV file contains time series data with a set number of samples per second.
# Each sample represents the amplitude of the audio signal at that specific time. In a 16-bit system, like the files in `mini_speech_commands`, the values range from -32768 to 32767.
# The sample rate for this dataset is 16kHz.
# Note that `tf.audio.decode_wav` will normalize the values to the range [-1.0, 1.0].
def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis=-1)


# Let's define a method that will take in the filename of the WAV file and output a tuple containing the audio and labels for supervised training.
def get_waveform_and_label(file_path):
    label = get_label(file_path)
    file_name = get_filename(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, label, file_name


# # Let's define a method that will take in the filename of the WAV file and output a tuple containing the audio and labels for supervised training.
# def get_waveform_and_label_and_filename(file_path):
#     label = get_label(file_path)
#     file_name = get_filename(file_path)
#     audio_binary = tf.io.read_file(file_path)
#     waveform = decode_audio(audio_binary)
#     return waveform, label, file_name


def get_commands(data_dir):
    # Sets wanted commands for training (Available commands: Down, Go, Left, No, Right, Stop, Up, Yes, and Unknown for commands that are not to be tested
    commands = np.array(tf.io.gfile.listdir(str(data_dir)))
    # commands = ["yes", "no", "unknown"] # --> Not needed
    return commands


def get_datasets_with_filenames(data_dir):
    commands = get_commands(data_dir=data_dir)
    print("Commands:", commands)

    # Extract the audio files into a list and shuffle it.
    filenames = tf.io.gfile.glob(str(data_dir) + "/*/*")
    filenames = tf.random.shuffle(sorted(filenames), seed=42)
    num_samples = len(filenames)
    print("Number of total examples:", num_samples)
    print(
        "Number of examples per label:",
        len(tf.io.gfile.listdir(str(data_dir / commands[0]))),
        len(tf.io.gfile.listdir(str(data_dir / commands[1]))),
        len(tf.io.gfile.listdir(str(data_dir / commands[2]))),
    )
    print("Example file tensor:", filenames[0])

    # Take 80% of total number examples for training set files
    train_files = filenames[:4249]
    # Take 10% of total number examples adding to 80% of total examples for validation set files
    val_files = filenames[4249 : 4249 + 531]
    # Take -10% of total number examples for test set files
    test_files = filenames[-531:]
    print("Training set size", len(train_files))
    print("Validation set size", len(val_files))
    print("Test set size", len(test_files))

    # You will now apply `process_path` to build your training set to extract the audio-label pairs and check the results.
    # You'll build the validation and test sets using a similar procedure later on.
    train_ds = preprocess_dataset(train_files)
    val_ds = preprocess_dataset(val_files)
    test_ds = preprocess_dataset(test_files)
    print(val_ds)
    print(test_ds)
    return train_ds, val_ds, test_ds


def remove_filename(audio, label, file_name):
    return audio, label


def get_datasets(data_dir):
    train_ds, val_ds, test_ds = get_datasets_with_filenames(data_dir=data_dir)
    return (
        train_ds.map(remove_filename, num_parallel_calls=AUTOTUNE),
        val_ds.map(remove_filename, num_parallel_calls=AUTOTUNE),
        test_ds.map(remove_filename, num_parallel_calls=AUTOTUNE),
    )


def quantize(array, scale, zero_point, dtype: np.dtype):
    input_min = np.iinfo(dtype).min
    input_max = np.iinfo(dtype).max
    return np.clip(np.floor(array / scale + zero_point), input_min, input_max)


def seed_tf_random():
    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        "--size",
        dest="size",
        type=int,
        action="store",
        required=False,
        default=1000,
        help="Generate subset of the dataset. Maximum size is 100000. Default size is 1000",
    )
    argparser.add_argument(
        "-o",
        "--output_dir",
        dest="output_dir",
        action="store",
        required=False,
        default="dataset",
        help="Output directory to store the dataset",
    )
    argparser.add_argument(
        "-m",
        "--model",
        dest="model",
        action="store",
        required=False,
        help="Path to model for which to generate dataset. It will extract the quantization parameters",
    )
    args = argparser.parse_args()

    print("TensorFlow Version: {}".format(tf.version.VERSION))
    seed_tf_random()

    prepare_dataset(data_dir=data_dir)
    train_ds, val_ds, test_ds = get_datasets_with_filenames(data_dir=data_dir)

    print("Training set size", len(train_ds))
    print("Validation set size", len(val_ds))
    print("Test set size", len(test_ds))

    is_input_quantized = False
    if args.model:
        tflite_interpreter = tf.lite.Interpreter(model_path=args.model)
        # Get input and output details
        input_details = tflite_interpreter.get_input_details()
        input_scale, input_zero_point = input_details[0]["quantization"]
        is_input_quantized = input_details[0]["dtype"] != np.float32
        input_dtype = input_details[0]["dtype"]
        print(f"Model has quantized input: {is_input_quantized}")
        print(
            f"Model's input quantization parameters: scale = {input_scale}, zero_point = {input_zero_point}"
        )

    # Generate input vectors for model:
    print("Generating {} binary tensors to: {}".format(args.size, args.output_dir))
    if os.path.isdir(args.output_dir):
        print("Output directory exists, removing!")
        shutil.rmtree(args.output_dir)
    os.mkdir(args.output_dir)

    for tensor, _, file_name in tqdm(test_ds.take(args.size)):
        tensor = tensor.numpy()
        if is_input_quantized:
            tensor = quantize(
                tensor, input_scale, input_zero_point, input_dtype
            ).astype(input_dtype)
        tensor.tofile(
            os.path.join(args.output_dir, file_name.numpy().decode() + ".bin")
        )

    with zipfile.ZipFile(args.output_dir + ".zip", mode="w") as archive:
        for i in os.listdir(args.output_dir):
            archive.write(os.path.join(args.output_dir, i), arcname=i)
