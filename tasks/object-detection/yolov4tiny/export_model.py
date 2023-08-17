# Copyright 2023 NXP
# SPDX-License-Identifier: BSD-3-Clause

import os
import numpy as np
import struct
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, LeakyReLU
from tensorflow.keras.layers import ZeroPadding2D, UpSampling2D
from tensorflow.keras.layers import MaxPool2D, add, concatenate
from tensorflow.keras.models import Model
import argparse
import PIL.Image as im
import random

random.seed(42)

N_CALIBRATION_IMAGES = 100

parser = argparse.ArgumentParser()
parser.add_argument('--weights_path', help='path to darknet weights')
parser.add_argument('--output_path', help='path to save tflite model')
parser.add_argument('--images_path',
                    help='path to representative images for quantization',
                    default=None)
args = parser.parse_args()


def _conv_block(inp, convs, skip=False):
    x = inp
    count = 0

    for conv in convs:
        if count == (len(convs) - 2) and skip:
            skip_connection = x
        count += 1

        if conv['stride'] > 1:
            x = ZeroPadding2D(((1, 0), (1, 0)),
                              name='zerop_' + str(conv['layer_idx']))(
                x)  # peculiar padding as darknet prefer left and top

        x = Conv2D(conv['filter'],
                   conv['kernel'],
                   strides=conv['stride'],
                   # peculiar padding as darknet prefer left and top
                   padding='valid' if conv['stride'] > 1 else 'same',
                   name='convn_' + str(conv['layer_idx']) \
                   if conv['bnorm'] else 'conv_' + str(conv['layer_idx']),
                   activation=None,
                   use_bias=True)(x)

        if conv['activ'] == 1:
            x = LeakyReLU(alpha=0.1, name='leaky_' + str(conv['layer_idx']))(x)

    return add([skip_connection, x],
               name='add_' + str(conv['layer_idx'] + 1)) if skip else x


def _split_block(input_layer, layer_idx):
    s = tf.split(input_layer,
                 num_or_size_splits=2,
                 axis=-1,
                 name=f"split_{layer_idx}")
    return s[1]


def make_yolov4_tiny_model():
    input_image = Input(shape=(416, 416, 3),
                        batch_size=1,
                        name='input_0')
    # Layer 0
    x = _conv_block(input_image, [{'filter': 32,
                                   'kernel': 3,
                                   'stride': 2,
                                   'bnorm': True,
                                   'activ': 1,
                                   'layer_idx': 0}])
    layer_0 = x
    # Layer 1
    x = _conv_block(x, [{'filter': 64,
                         'kernel': 3,
                         'stride': 2,
                         'bnorm': True,
                         'activ': 1,
                         'layer_idx': 1}])
    layer_1 = x
    # Layer  2, concat1
    x = _conv_block(x, [{'filter': 64,
                         'kernel': 3,
                         'stride': 1,
                         'bnorm': True,
                         'activ': 1,
                         'layer_idx': 2}])
    layer_2 = x
    # Layer  3, route group
    x = _split_block(x, layer_idx=3)
    # Layer 4, concat_route_1
    x = _conv_block(x, [{'filter': 32,
                         'kernel': 3,
                         'stride': 1,
                         'bnorm': True,
                         'activ': 1,
                         'layer_idx': 4}])
    layer_4 = x
    # Layer 5, concat_route_2
    x = _conv_block(x, [{'filter': 32,
                         'kernel': 3,
                         'stride': 1,
                         'bnorm': True,
                         'activ': 1,
                         'layer_idx': 5}])
    layer_5 = x
    # Layer 6, concat route
    x = concatenate([layer_5, layer_4], axis=-1, name='concat_6')
    # Layer 7, concat2
    x = _conv_block(x, [{'filter': 64,
                         'kernel': 1,
                         'stride': 1,
                         'bnorm': True,
                         'activ': 1,
                         'layer_idx': 7}])
    layer_7 = x
    # Layer 8, concat
    x = concatenate([layer_2, layer_7], axis=-1, name='concat_8')
    # Layer 9
    x = MaxPool2D(pool_size=(2, 2), padding='same', name='layer_9')(x)

    # Layer 10, concat 1
    x = _conv_block(x, [{'filter': 128,
                         'kernel': 3,
                         'stride': 1,
                         'bnorm': True,
                         'activ': 1,
                         'layer_idx': 10}])
    layer_10 = x
    # Layer 11
    x = _split_block(x, layer_idx=11)
    # Layer 12, concat route 1
    x = _conv_block(x, [{'filter': 64,
                         'kernel': 3,
                         'stride': 1,
                         'bnorm': True,
                         'activ': 1,
                         'layer_idx': 12}])
    layer_12 = x
    # Layer 13, concat route 2
    x = _conv_block(x, [{'filter': 64,
                         'kernel': 3,
                         'stride': 1,
                         'bnorm': True,
                         'activ': 1,
                         'layer_idx': 13}])
    layer_13 = x
    # Layer 14
    x = concatenate([layer_13, layer_12], axis=-1, name='concat_14')
    # Layer 15, concat 2
    x = _conv_block(x, [{'filter': 128,
                         'kernel': 1,
                         'stride': 1,
                         'bnorm': True,
                         'activ': 1,
                         'layer_idx': 15}])
    layer_15 = x
    # Layer 16
    x = concatenate([layer_10, layer_15], axis=-1, name='concat_16')
    # Layer 17
    x = MaxPool2D(pool_size=(2, 2), padding='same', name='layer_17')(x)

    # Layer 18, concat 1
    x = _conv_block(x, [{'filter': 256,
                         'kernel': 3,
                         'stride': 1,
                         'bnorm': True,
                         'activ': 1,
                         'layer_idx': 18}])
    layer_18 = x
    # Layer 19
    x = _split_block(x, layer_idx=19)
    # Layer 20, concat route 1
    x = _conv_block(x, [{'filter': 128,
                         'kernel': 3,
                         'stride': 1,
                         'bnorm': True,
                         'activ': 1,
                         'layer_idx': 20}])
    layer_20 = x
    # Layer 21, concat route 2
    x = _conv_block(x, [{'filter': 128,
                         'kernel': 3,
                         'stride': 1,
                         'bnorm': True,
                         'activ': 1,
                         'layer_idx': 21}])
    layer_21 = x
    # Layer 22
    x = concatenate([layer_21, layer_20], axis=-1, name='concat_22')
    # Layer 23, concat 2, output 1 of cspdarknet
    x = _conv_block(x, [{'filter': 256,
                         'kernel': 1,
                         'stride': 1,
                         'bnorm': True,
                         'activ': 1,
                         'layer_idx': 23}])
    layer_23 = x
    # Layer 24
    x = concatenate([layer_18, layer_23], axis=-1, name='concat_24')
    # Layer 25
    x = MaxPool2D(pool_size=(2, 2), padding='same', name='layer_25')(x)

    # Layer 26, output 2 of cspdarknet
    x = _conv_block(x, [{'filter': 512,
                         'kernel': 3,
                         'stride': 1,
                         'bnorm': True,
                         'activ': 1,
                         'layer_idx': 26}])
    layer_26 = x

    # After backbone
    # Layer 27, concat 1, branch 1
    x = _conv_block(layer_26, [{'filter': 256,
                                'kernel': 1,
                                'stride': 1,
                                'bnorm': True,
                                'activ': 1,
                                'layer_idx': 27}])
    layer_27 = x

    # Layer 28
    x = _conv_block(x, [{'filter': 512,
                         'kernel': 3,
                         'stride': 1,
                         'bnorm': True,
                         'activ': 1,
                         'layer_idx': 28}])
    layer_28 = x
    # Layer 29, output of large grid
    x = _conv_block(x, [{'filter': 255,
                         'kernel': 1,
                         'stride': 1,
                         'bnorm': True,
                         'activ': 0,
                         'layer_idx': 29}])
    layer_29 = x

    # Layer 30, continue from layer_27
    x = _conv_block(layer_27, [{'filter': 128,
                                'kernel': 1,
                                'stride': 1,
                                'bnorm': True,
                                'activ': 1,
                                'layer_idx': 30}])
    layer_30 = x
    # Layer 31
    x = UpSampling2D(size=(2, 2),
                     name='upsamp_31',
                     interpolation='bilinear')(x)
    layer_31 = x
    # Layer 32
    x = concatenate([layer_31, layer_23], axis=-1, name='concat_32')
    # Layer 33
    x = _conv_block(x, [{'filter': 256,
                         'kernel': 3,
                         'stride': 1,
                         'bnorm': True,
                         'activ': 1,
                         'layer_idx': 33}])
    # Layer 34, output of medium grid
    x = _conv_block(x, [{'filter': 255,
                         'kernel': 1,
                         'stride': 1,
                         'bnorm': True,
                         'activ': 0,
                         'layer_idx': 34}])
    layer_34 = x

    # End
    model = Model(input_image, [layer_34, layer_29], name='Yolov4-tiny')
    model.summary()
    return model


# Define the model
model = make_yolov4_tiny_model()

model.summary()


# load weights in keras

class WeightReader:
    def __init__(self, weight_file):
        with open(weight_file, 'rb') as w_f:
            major, = struct.unpack('i', w_f.read(4))
            minor, = struct.unpack('i', w_f.read(4))
            revision, = struct.unpack('i', w_f.read(4))

            if (major * 10 + minor) >= 2 and major < 1000 and minor < 1000:
                print("reading 64 bytes")
                w_f.read(8)
            else:
                print("reading 32 bytes")
                w_f.read(4)

            transpose = (major > 1000) or (minor > 1000)

            binary = w_f.read()

        self.offset = 0
        self.all_weights = np.frombuffer(binary, dtype='float32')
        print(f"weight total length {len(self.all_weights)}")

    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset - size:self.offset]

    def load_weights(self, model):
        count = 0
        ncount = 0
        for i in range(35):
            try:

                conv_layer = model.get_layer('convn_' + str(i))

                filter = conv_layer.kernel.shape[-1]
                # kernel*kernel*c*filter
                nweights = np.prod(conv_layer.kernel.shape)

                print(f"loading weights of convolution #" +
                      str(i) + "- nb parameters: " +
                      str(nweights + filter))

                if i in [29, 34]:
                    bias = self.read_bytes(filter)  # bias
                    weights = self.read_bytes(nweights)  # weights

                else:
                    bias = self.read_bytes(filter)  # bias
                    scale = self.read_bytes(filter)  # scale
                    mean = self.read_bytes(filter)  # mean
                    var = self.read_bytes(filter)  # variance
                    weights = self.read_bytes(nweights)  # weights

                    # normalize bias
                    bias = bias - scale * mean / (np.sqrt(var + 0.00001))

                    # normalize weights
                    weights = np.reshape(weights,
                                         (filter, int(nweights / filter)))
                    A = scale / (np.sqrt(var + 0.00001))
                    A = np.expand_dims(A, axis=0)
                    weights = weights * A.T
                    weights = np.reshape(weights, (nweights))

                shp = list(reversed(conv_layer.get_weights()[0].shape))
                weights = weights.reshape(shp)
                weights = weights.transpose([2, 3, 1, 0])

                if len(conv_layer.get_weights()) > 1:
                    a = conv_layer.set_weights([weights, bias])
                else:
                    a = conv_layer.set_weights([weights])

                count = count + 1
                ncount = ncount + nweights + filter

            except ValueError:
                print("no convolution #" + str(i))

        print(count,
              "Convolution Normalized Layers are loaded with ",
              ncount,
              " parameters")

    def reset(self):
        self.offset = 0


darknet_model = args.weights_path + '/yolov4-tiny.weights'
weight_reader = WeightReader(darknet_model)
weight_reader.load_weights(model)


def image_resize(image, resize_shape):
    image_copy = np.copy(image)
    resize_h, resize_w = resize_shape
    orig_h, orig_w, _ = image_copy.shape

    scale = min(resize_h / orig_h, resize_w / orig_w)
    temp_w, temp_h = int(scale * orig_w), int(scale * orig_h)
    image_resized = image.resize((temp_w, temp_h), im.BILINEAR)
    image_paded = np.full(shape=[resize_h, resize_w, 3], fill_value=128.0)
    r_w = (resize_w - temp_w) // 2  # real_w
    r_h = (resize_h - temp_h) // 2  # real_h
    image_paded[r_h:temp_h + r_h, r_w:temp_w + r_w, :] = image_resized
    image_paded = image_paded / 255.
    return image_paded


def representative_dataset():
    _, h, w, _ = model.input_shape
    image_folder = args.images_path
    image_files = os.listdir(image_folder)
    random.shuffle(image_files)
    image_files = image_files[:N_CALIBRATION_IMAGES]
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        original_image = im.open(image_path)
        if original_image.mode != "RGB":
            continue
        image_data = image_resize(original_image, [h, w])
        img_in = image_data[np.newaxis, ...].astype(np.float32)
        yield [img_in]


def dummy_dataset():
    _, h, w, _ = model.input_shape
    for i in range(N_CALIBRATION_IMAGES):
        # Tensorflow basic format : NHWC
        img_in = np.random.randn(1, h, w, 3).astype('float32')
        yield [img_in]


converter = tf.lite.TFLiteConverter.from_keras_model(model)

# quantized model
tflite_quant = args.output_path + '/yolov4-tiny_416_quant.tflite'

converter.optimizations = [tf.lite.Optimize.DEFAULT]
if args.images_path is not None:
    converter.representative_dataset = representative_dataset
else:  # Dummy dataset if no representative dataset is given
    converter.representative_dataset = dummy_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.float32

tflite_model = converter.convert()
with open(tflite_quant, 'wb') as f:
    f.write(tflite_model)


# float32 model
converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_float = args.output_path + '/yolov4-tiny_416_float32.tflite'

tflite_model = converter.convert()
with open(tflite_float, 'wb') as f:
    f.write(tflite_model)
