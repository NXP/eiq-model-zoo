#!/usr/bin/env python3
# Copyright 2024 NXP
# SPDX-License-Identifier: MIT

import numpy as np
from utils import load_image
from glob import glob

imgs = glob('lfw/*/*jpg')

imgs = [load_image(i)[1][0, ...] for i in imgs]

batch = np.stack(imgs, 0)
np.save("calib.npy", batch)
