#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright 2023 NXP

import os
import argparse


def parse_command():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--background_volume",
        type=float,
        default=0.1,
        help="""\
      How loud the background noise should be, between 0 and 1.
      """,
    )
    parser.add_argument(
        "--background_frequency",
        type=float,
        default=0.8,
        help="""\
      How many of the training samples have background noise mixed in.
      """,
    )

    parser.add_argument(
        "--clip_duration_ms",
        type=int,
        default=1000,
        help="Expected duration in milliseconds of the wavs",
    )
    parser.add_argument(
        "--window_size_ms",
        type=float,
        default=30.0,
        help="How long each spectrogram timeslice is",
    )
    parser.add_argument(
        "--window_stride_ms",
        type=float,
        default=20.0,
        help="How long each spectrogram timeslice is",
    )
    parser.add_argument(
        "--feature_type",
        type=str,
        default="mfcc",
        choices=["mfcc", "lfbe", "td_samples"],
        help='Type of input features. Valid values: "mfcc" (default), "lfbe", "td_samples"',
    )
    parser.add_argument(
        "--dct_coefficient_count",
        type=int,
        default=10,
        help="How many MFCC or log filterbank energy features",
    )

    parser.add_argument(
        "--model_architecture",
        type=str,
        default="ds_cnn",
        help="What model architecture to use",
    )

    parser.add_argument(
        "--tfl_file_name",
        default="kws_ref_model.tflite",
        help="File name to which the TF Lite model will be saved (quantize.py) or loaded (eval_quantized_model)",
    )

    parser.add_argument(
        "--file",
        type=str,
        required=True,
        default='example_input.wav',
        help=""""\
      Input wav file for inference.
      """,
    )

    Flags, unparsed = parser.parse_known_args()
    return Flags, unparsed
