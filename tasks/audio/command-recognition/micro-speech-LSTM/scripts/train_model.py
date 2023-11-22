#!/usr/bin/env python

# SPDX-License-Identifier: MIT
# Copyright 2023 NXP

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import dataset
import model_utils

EPOCH = 100
UNROLL = False

# Set seed for experiment reproducibility
dataset.seed_tf_random()

print("TensorFlow Version: {}".format(tf.version.VERSION))
print("Will train minispeech LSTM model")
dataset.prepare_dataset(data_dir=dataset.data_dir)
train_ds, val_ds, test_ds = dataset.get_datasets(data_dir=dataset.data_dir)

# Batch the training and validation sets for model training.
batch_size = 64
train_ds = train_ds.batch(batch_size)
val_ds = val_ds.batch(batch_size)


train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

model = model_utils.get_model(unroll=UNROLL)
model.summary()

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

EPOCHS = 100
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=EPOCHS * 0.7),
)

# Let's check the training and validation loss curves to see how your model has improved during training.
metrics = history.history
plt.plot(history.epoch, metrics["loss"], metrics["val_loss"])
plt.legend(["loss", "val_loss"])
plt.show()

model.save(model_utils.MODEL_PATH)
