#!/usr/bin/env bash
# Copyright 2022-2025 NXP
# SPDX-License-Identifier: MIT

set -e
source ./env/bin/activate

#Download the fw.bin from the faces_emore repo
gdown 1SXS4-Am3bsKSK615qbYdbA_FMVh3sAvR

unzip faces_emore.zip

source ./env/bin/activate
python3.8 -c "
import Evaluation
eval = Evaluation.eval_callback('facenet512_uint8.tflite','faces_emore/lfw.bin', batch_size=1, PCA_acc=True)
eval.on_epoch_end(0)
"

rm -rf faces_emore.zip
