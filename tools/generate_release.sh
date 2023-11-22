#!/usr/bin/env bash
# Copyright 2022-2023 NXP
# SPDX-License-Identifier: MIT

set -e

RECIPES=$(find . -name recipe.sh)

ORIG_DIR=$(pwd)

for recipe in $RECIPES; do
    (
    cd "$(dirname "${recipe}")" || exit
    bash recipe.sh
    )
done

TFLITE_FILES=$(find . -name "*.tflite")

tar czf release_$(date -I).tar.gz $TFLITE_FILES
