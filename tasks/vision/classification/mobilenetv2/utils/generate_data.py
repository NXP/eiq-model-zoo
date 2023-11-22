#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright 2023 NXP

import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
from tqdm import tqdm
from PIL import Image


h, w = 224, 224

def prepare_image(image):
    image = np.array(image['image'])
    image = Image.fromarray(image)
    image = image.resize((h, w))
    tensor = np.array(image, dtype=np.float32) / 255.
    tensor = np.expand_dims(tensor, axis=0)
    return tensor

def inference(interpreter, tensor):

    input_details = interpreter.get_input_details()

    if tensor.shape[-1] != 3:
        return None

    output_details = interpreter.get_output_details()
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]["index"], tensor)
    interpreter.invoke()

    pred_y = interpreter.get_tensor(output_details[0]["index"])
    return pred_y.argmax()

if __name__ == "__main__":
    ds, ds_info = tfds.load('imagenet_v2/topimages', split='test', with_info=True)

    model = tf.keras.applications.MobileNetV2(input_shape=(h, w, 3))

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.input_shape = (1, h, w, 3)
    tflite_quant_model = converter.convert()

    with open("mobilenet_v2_float32.tflite", "wb") as f:
        f.write(tflite_quant_model)

    correct_imgs = np.empty((1, h, w, 3), dtype=np.float32)


    for file in tqdm(ds.take(2000)):
        y_true = file['label'].numpy()
        tensor = prepare_image(file)
        interpreter = tf.lite.Interpreter(model_content=tflite_quant_model)
        pred_y = inference(interpreter, tensor)
        if not pred_y:
            continue
        if y_true == pred_y:
            correct_imgs = np.append(correct_imgs, tensor, axis=0)

    np.save('quantization_data.npy', correct_imgs)
