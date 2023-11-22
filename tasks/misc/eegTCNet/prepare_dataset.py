#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright 2023 NXP

import argparse
import fnmatch
import glob
import importlib
import os.path
import shutil

import numpy as np
from tqdm import tqdm

data_loading = importlib.import_module("eeg-tcnet-master.utils.data_loading")


def prepare_data(path):
    (
        X_train,
        _,
        y_train_onehot,
        X_test,
        _,
        y_test_onehot,
    ) = data_loading.prepare_features(path, 0, False)
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    data = np.concatenate([X_train, X_test], axis=0)
    return data


def load_data(item):
    item = item.astype(np.int8)
    return item.tobytes()



if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        "--size",
        dest="size",
        type=int,
        action="store",
        required=False,
        default=100,
        help="Generate subset of the dataset. Maximum size is 1000. Default size is 100",
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
        "--path",
        # action="store",
        required=True,
        help="Directory, where data are located."
    )

    args = argparser.parse_args()

    BASE_DIR = os.path.join(os.getcwd(), args.path, '')
    print("Generating {} binary tensors to: {}".format(args.size, args.output_dir))
    if os.path.isdir(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.mkdir(args.output_dir)

    data = np.empty((1, 1, 22, 1125))
    filename = ""

    for file in fnmatch.filter(os.listdir(BASE_DIR), '*.mat'):
        data = np.concatenate([data, prepare_data(BASE_DIR)], axis=0)
        if filename == "":
            filename = file.split(".")[0]

    i = 0
    for item in tqdm(data[: args.size]):
        output_filename = filename + "_" + str(i) + ".bin"
        with open(os.path.join(args.output_dir, output_filename), "wb") as f:
            f.write(load_data(item))
        i += 1
    print("Finished")
