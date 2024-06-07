#!/usr/bin/env python3
# Copyright 2024 NXP
# SPDX-License-Identifier: MIT

import cv2
import tensorflow as tf
import numpy as np
import time
import random

OBJECT_DETECTOR_TFLITE = 'centernet.tflite'
LABELS_FILE = 'coco-labels-2014_2017.txt'
IMAGE_FILENAME = 'example_input.jpg'

SCORE_THRESHOLD = 0.4
NMS_IOU_THRESHOLD = 0.5
INFERENCE_IMG_SIZE = 512
MAX_DETS = 100

with open(LABELS_FILE, 'r') as f:
    COCO_CLASSES = [line.strip() for line in f.readlines()]

interpreter = tf.lite.Interpreter(OBJECT_DETECTOR_TFLITE)
interpreter.allocate_tensors()


def gen_box_colors():
    colors = []
    for _ in range(len(COCO_CLASSES)):
        r = random.randint(100, 255)
        g = random.randint(100, 255)
        b = random.randint(100, 255)
        colors.append((r, g, b))

    return colors


BOX_COLORS = gen_box_colors()


def load_image(filename):
    orig_image = cv2.imread(filename, 1)
    image = orig_image
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (INFERENCE_IMG_SIZE, INFERENCE_IMG_SIZE))
    image = np.expand_dims(image, axis=0)
    return orig_image, image


def mask_from_true_image_shape(data_shape, true_image_shapes):
    """Get a binary mask based on the true_image_shape.
       Original function from the tensorflow implementation of this
       model from the tensorflow model garden.
    """

    mask_h = tf.cast(
        tf.range(data_shape[1]) < true_image_shapes[:, tf.newaxis, 0],
        tf.float32)
    mask_w = tf.cast(
        tf.range(data_shape[2]) < true_image_shapes[:, tf.newaxis, 1],
        tf.float32)
    mask = tf.expand_dims(
        mask_h[:, :, tf.newaxis] * mask_w[:, tf.newaxis, :], 3)
    return mask


def _multi_range(limit,
                 value_repetitions=1,
                 range_repetitions=1,
                 dtype=tf.int32):
    """Creates a sequence with optional value duplication and range repetition.
       Original function from the tensorflow implementation of this
       model from the tensorflow model garden.
    """
    return tf.reshape(
        tf.tile(
            tf.expand_dims(tf.range(limit, dtype=dtype), axis=-1),
            multiples=[range_repetitions, value_repetitions]), [-1])


def prediction_tensors_to_boxes(y_indices, x_indices, height_width_predictions,
                                offset_predictions):
    """Converts CenterNet class-center, offset and size predictions to boxes.
       Original function from the tensorflow implementation of this
       model from the tensorflow model garden.
    """
    batch_size, num_boxes = y_indices.shape
    _, height, width, _ = height_width_predictions.shape
    height, width = tf.cast(height, tf.float32), tf.cast(width, tf.float32)

    # TF Lite does not support tf.gather with batch_dims > 0, so we need to use
    # tf_gather_nd instead and here we prepare the indices for that.
    combined_indices = tf.stack([
        _multi_range(batch_size, value_repetitions=num_boxes),
        tf.reshape(y_indices, [-1]),
        tf.reshape(x_indices, [-1])
    ], axis=1)
    new_height_width = tf.gather_nd(height_width_predictions, combined_indices)
    new_height_width = tf.reshape(new_height_width, [batch_size, num_boxes, 2])

    new_offsets = tf.gather_nd(offset_predictions, combined_indices)
    offsets = tf.reshape(new_offsets, [batch_size, num_boxes, 2])

    y_indices = tf.cast(y_indices, tf.float32)
    x_indices = tf.cast(x_indices, tf.float32)

    height_width = tf.maximum(new_height_width, 0)
    heights, widths = tf.unstack(height_width, axis=2)
    y_offsets, x_offsets = tf.unstack(offsets, axis=2)

    ymin = y_indices + y_offsets - heights / 2.0
    xmin = x_indices + x_offsets - widths / 2.0
    ymax = y_indices + y_offsets + heights / 2.0
    xmax = x_indices + x_offsets + widths / 2.0

    ymin = tf.clip_by_value(ymin, 0., height)
    xmin = tf.clip_by_value(xmin, 0., width)
    ymax = tf.clip_by_value(ymax, 0., height)
    xmax = tf.clip_by_value(xmax, 0., width)
    boxes = tf.stack([ymin, xmin, ymax, xmax], axis=2)

    return boxes, combined_indices


def convert_strided_predictions_to_normalized_boxes(boxes, stride,
                                                    true_image_shapes):
    """Converts predictions in the output space to normalized boxes.
       Original function from the tensorflow implementation of this
       model from the tensorflow model garden.
    """
    # Note: We use tf ops instead of functions in box_list_ops to make this
    # function compatible with dynamic batch size.
    boxes = boxes * stride
    true_image_shapes = tf.tile(
        true_image_shapes[:, tf.newaxis, :2], [1, 1, 2])
    boxes = boxes / tf.cast(true_image_shapes, tf.float32)
    boxes = tf.clip_by_value(boxes, 0.0, 1.0)
    return boxes


def topK(heatmap, K=10):
    """Custom implementation of the topK algorithm for this specific model
    """
    batch, _, w, n_classes = heatmap.shape
    heatmap = tf.reshape(heatmap, [batch, -1])
    safe_k = tf.minimum(K, tf.shape(heatmap)[1])
    topk_scores, topk_indices = tf.math.top_k(heatmap, safe_k)
    topk_classes = topk_indices // n_classes
    ys = topk_classes // w
    xs = topk_classes - ys * w
    class_indices = topk_indices - topk_classes * n_classes
    return topk_scores, class_indices, ys, xs


def decode_output(heatmap, size, offset, output_stride=4, K=100):
    """First decoding function. Corresponds to the post-processing steps
    of the original model with their nms implementation removed.
    """
    heatmap = tf.sigmoid(heatmap)
    _, h, w, _ = heatmap.shape
    true_image_shape = np.array([[h * output_stride, w * output_stride, 3]])
    topk_scores, class_indices, ys, xs = topK(heatmap, K)

    detections_unormalized, _ = prediction_tensors_to_boxes(
        ys, xs, size, offset)
    detections = detections_unormalized
    detections = convert_strided_predictions_to_normalized_boxes(
        detections_unormalized, output_stride, true_image_shape)
    return detections, topk_scores, class_indices


def run_inference(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    type_tf = input_details[0]['dtype']
    if type_tf == np.int8:
        input_scale, input_zero_point = input_details[0]["quantization"]
        image = image * input_scale - input_zero_point
        image = image.astype(np.int8)
    else:
        image = image.astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    size = interpreter.get_tensor(output_details[2]['index'])
    offset = interpreter.get_tensor(output_details[4]['index'])
    heatmap = interpreter.get_tensor(output_details[0]['index'])
    processed_image = interpreter.get_tensor(output_details[3]['index'])
    return heatmap, size, offset, processed_image


def decode_output_nms(scores, boxes, classes,
                      score_threshold=SCORE_THRESHOLD,
                      iou_threshold=NMS_IOU_THRESHOLD):

    # apply original NMS from tensorflow
    inds = tf.image.non_max_suppression(boxes, scores, MAX_DETS,
                                        score_threshold=score_threshold,
                                        iou_threshold=iou_threshold)

    # keep only selected boxes
    boxes = tf.gather(boxes, inds)
    scores = tf.gather(scores, inds)
    classes = tf.gather(classes, inds)

    return boxes, scores, classes


if __name__ == "__main__":
    orig_image, processed_image = load_image(IMAGE_FILENAME)

    start = time.time()
    heatmap, size, offset, processed_image = run_inference(
        interpreter, processed_image)
    end = time.time()

    boxes, scores, classes = decode_output(heatmap, size, offset)
    boxes = tf.squeeze(boxes)
    scores = tf.squeeze(scores)
    classes = tf.squeeze(classes)

    # apply tensorflow's implementation of nms to the outputs.
    boxes, scores, classes = decode_output_nms(scores, boxes, classes,
                                               score_threshold=SCORE_THRESHOLD,
                                               iou_threshold=NMS_IOU_THRESHOLD)

    boxes = tf.reshape(boxes, [-1, 4])
    boxes = boxes.numpy()
    print("Inference time", end - start, "ms")
    print("Detected", boxes.shape[0], "object(s)")
    print("Box coordinates:")
    for i in range(boxes.shape[0]):
        box = boxes[i, :]
        class_name = COCO_CLASSES[classes[i].numpy().astype(int)]
        score = scores[i].numpy()
        color = BOX_COLORS[classes[i].numpy().astype(int)]
        shp = orig_image.shape

        # the boxes have to be projected onto the original image's dimension
        orig_h, orig_w = shp[0], shp[1]
        ymin = int(max(1, (box[0] * orig_h)))
        xmin = int(max(1, (box[1] * orig_w)))
        ymax = int(min(orig_h, (box[2] * orig_h)))
        xmax = int(min(orig_w, (box[3] * orig_w)))
        new_box = tf.stack([ymin, xmin, ymax, xmax], axis=-1)
        print(new_box, end=" ")
        print(classes[i])
        print("class", class_name, end=" ")
        print("score", score)
        cv2.rectangle(orig_image, (xmin, ymin), (xmax, ymax), color, 3)
        cv2.putText(orig_image, f"{class_name} {score:.2f}",
                    (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imwrite('example_output.jpg', orig_image)
    cv2.imshow('', orig_image)
    cv2.waitKey()
