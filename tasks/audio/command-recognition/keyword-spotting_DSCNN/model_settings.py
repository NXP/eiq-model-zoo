#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright 2023 NXP


def prepare_model_settings(label_count, sample_rate, args):
    """Calculates common settings needed for all models.
    Args:
      label_count: How many classes are to be recognized.
      sample_rate: Number of audio samples per second.
      clip_duration_ms: Length of each audio clip to be analyzed.
      window_size_ms: Duration of frequency analysis window.
      window_stride_ms: How far to move in time between frequency windows.
      dct_coefficient_count: Number of frequency bins to use for analysis.
    Returns:
      Dictionary containing common settings.
    """
    desired_samples = int(sample_rate * args.clip_duration_ms / 1000)
    if args.feature_type == "td_samples":
        window_size_samples = 1
        spectrogram_length = desired_samples
        dct_coefficient_count = 1
        window_stride_samples = 1
        fingerprint_size = desired_samples
    else:
        dct_coefficient_count = args.dct_coefficient_count
        window_size_samples = int(sample_rate * args.window_size_ms / 1000)
        window_stride_samples = int(sample_rate * args.window_stride_ms / 1000)
        length_minus_window = desired_samples - window_size_samples
        if length_minus_window < 0:
            spectrogram_length = 0
        else:
            spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
            fingerprint_size = args.dct_coefficient_count * spectrogram_length
    return {
        "desired_samples": desired_samples,
        "window_size_samples": window_size_samples,
        "window_stride_samples": window_stride_samples,
        "feature_type": args.feature_type,
        "spectrogram_length": spectrogram_length,
        "dct_coefficient_count": dct_coefficient_count,
        "fingerprint_size": fingerprint_size,
        "label_count": label_count,
        "sample_rate": sample_rate,
        "background_frequency": args.background_frequency,
        "background_volume_range_": 0.1,
    }
