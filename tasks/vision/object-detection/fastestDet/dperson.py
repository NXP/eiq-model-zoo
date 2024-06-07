#!/usr/bin/env python3
# Copyright 2022-2024 NXP
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

def letterbox_resize(img, shape):
    """
    Function adapted from:
    https://medium.com/mlearning-ai/letterbox-in-object-detection-77ee14e5ac46
    img:         input image in numpy array
    input_shape: (height, width) of input image, this is the target shape for the model
    """
    img_h, img_w, _ = img.shape
    new_h, new_w = shape[0], shape[1]
    offset_h, offset_w = 0, 0
    if (new_w / img_w) <= (new_h / img_h):
        new_h = int(img_h * new_w / img_w)
        offset_h = (shape[0] - new_h) // 2
    else:
        new_w = int(img_w * new_h / img_h)
        offset_w = (shape[1] - new_w) // 2
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    img = np.full((shape[0], shape[1], 3), 150, dtype=np.uint8)

    img[offset_h: (offset_h + new_h), offset_w: (offset_w + new_w), :] = resized
    return img, new_h, new_w

# sigmoid function
def sigmoid(x):
    return 1. / (1 + np.exp(-x))
# tanh function
def tanh(x):
    return 2. / (1 + np.exp(-2 * x)) - 1

class dperson:
    def __init__(self, tflite_model_path):
        self.interpreter = tflite.Interpreter(model_path=tflite_model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_shape =  self.input_details[0]["shape"]
        self.input_idx = self.input_details[0]["index"]
        self.output_idx = self.output_details[0]["index"]
        self.bbox = None

        # size of the image after letterbox resizing, without padding
        self.prepad_h = None
        self.prepad_w = None

        # padding offset for letterbox resizing
        self.offset_h = None
        self.offset_w = None

    def quantize(
        self,
        data,
        details,
        verbose=True,
    ):
        """quantizes the data based on data details"""
        for info in details:
            name = info['name']
            dtype = info['dtype']
            for k, v in info.items():
                if verbose:
                    print(f'{name}.input_details.{k}: {v}')
            if dtype != np.float32:
                scales = info['quantization_parameters']['scales']
                zero_points = info['quantization_parameters']['zero_points']
                data = data / scales + zero_points
            data = data.astype(dtype)
            if verbose:
                print(f"quantizing data {name} to {dtype.__name__}")
        return data

    def dequantize(
        self,
        data,
        details,
        verbose=True,
    ):
        """dequantizes the data based on data details"""
        for info in details:
            name = info['name']
            dtype = info['dtype']
            for k, v in info.items():
                if verbose:
                    print(f'{name}.output_details.{k}: {v}')
            if dtype != np.float32:
                scales = info['quantization_parameters']['scales']
                zero_points = info['quantization_parameters']['zero_points']
                data = (data.astype(dtype) - zero_points) * scales
                if dtype == np.int8:
                    # ignore '_int8' suffix of name
                    name = name
                if verbose:
                    print(f"dequantizing data {name} to {dtype.__name__}")
        return data

    def preprocess(self, img):
        # letterbox resizing to preserve aspect ratio
        img, self.prepad_h, self.prepad_w = letterbox_resize(
            img,
            (self.input_shape[1], self.input_shape[2]),
        )
        # store shapes for bbox postprocessing
        self.offset_h = (self.input_shape[1] - self.prepad_h) // 2
        self.offset_w = (self.input_shape[2] - self.prepad_w) // 2

        img = img.reshape((1, 220, 220, 3)) / 255
        return img.astype('float32')

    def nms(self, bbox, thresh=0.4):
        if len(bbox) > 0:
            # bbox: N*M, where N is the number of the bounding boxes, (x1, y1, x2, y2, score)
            x1 = bbox[:, 0]
            y1 = bbox[:, 1]
            x2 = bbox[:, 2]
            y2 = bbox[:, 3]
            scores = bbox[:, 4]
            areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # the area of each box
            order = scores.argsort()[::-1]  # sort the bounding box according to score
            keep = []  # save the bounding boxes
            while order.size > 0:
                i = order[0]  # save the bbox of highest score without limitation
                keep.append(i)
                # compute the cross area between the current bbox and the others
                xx1 = np.maximum(x1[i], x1[order[1:]])
                yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]])
                yy2 = np.minimum(y2[i], y2[order[1:]])
                w = np.maximum(0.0, xx2 - xx1 + 1)
                h = np.maximum(0.0, yy2 - yy1 + 1)
                inter = w * h
                # compute the percentage cross area
                ovr = inter / (areas[i] + areas[order[1:]] - inter)
                # save the one whose cross area is lower than threshold
                order = order[np.where(ovr <= thresh)[0] + 1]
            # output = []
            # for i in keep:
            #     output.append(bbox[i].tolist())
            return bbox[keep].tolist()
        else:
            return bbox

    def detect(self, img, thresh):
        img_h, img_w, _ = img.shape

        data = self.preprocess(img)

        # model inference
        data = self.quantize(data, self.input_details)
        self.interpreter.set_tensor(self.input_idx, data)
        self.interpreter.invoke()
        feature_map = self.interpreter.get_tensor(self.output_idx)
        feature_map = self.dequantize(feature_map, self.output_details)[0]

        # get the height and width of the feature map
        fmh = feature_map.shape[0]
        fmw = feature_map.shape[1]
        self.bbox = []
        # post-process for feature map
        for h in range(fmh):
            for w in range(fmw):
                data = feature_map[h][w]
                # the confidence of the bounding boxes
                obj_score, cls_score = data[0], data[5:].max()
                score = (obj_score ** 0.6) * (cls_score ** 0.4)
                if score > thresh:
                    # the category of the class
                    cls_index = np.argmax(data[5:])
                    # the offset of the center location
                    x_offset, y_offset = tanh(data[1]), tanh(data[2])
                    # the normalization of the width and height
                    box_width, box_height = sigmoid(data[3]), sigmoid(data[4])

                    # the center of the box
                    box_cx = (w + x_offset) / fmw
                    box_cy = (h + y_offset) / fmh

                    # cx,cy,w,h => x1, y1, x2, y2
                    x1, y1 = box_cx - 0.5 * box_width, box_cy - 0.5 * box_height
                    x2, y2 = box_cx + 0.5 * box_width, box_cy + 0.5 * box_height

                    # adjust to original image resolution
                    x1, y1 = int(x1 * self.input_shape[2]), int(y1 * self.input_shape[1])
                    x2, y2 = int(x2 * self.input_shape[2]), int(y2 * self.input_shape[1])

                    # adjust box coordinates to compensate for letterbox resizing
                    x1, y1, x2, y2 = self.adjust_bboxes(x1, y1, x2, y2)

                    x1, y1, x2, y2 = int(x1 * img_w), int(y1 * img_h), int(x2 * img_w), int(y2 * img_h)
                    self.bbox.append([x1, y1, x2, y2, score, cls_index])
        self.bbox = self.nms(np.array(self.bbox))
        return self.bbox

    def adjust_bboxes(self, x1, y1, x2, y2):
        box_width = x2 - x1
        box_height = y2 - y1
        cx = (x1 + (x2 - x1) / 2) - self.offset_w
        cy = (y1 + (y2 - y1) / 2) - self.offset_h
        x1 = max(0, int(cx - box_width * 0.5)) / self.prepad_w
        y1 = max(0, int(cy - box_height * 0.5)) /  self.prepad_h
        x2 = min(self.input_shape[2], int(cx + box_width * 0.5)) / self.prepad_w
        y2 = min(self.input_shape[1], int(cy + box_height * 0.5)) / self.prepad_h
        return x1, y1, x2, y2

    def show(self, img):
        font = cv2.FONT_HERSHEY_SIMPLEX # font
        fontScale = 1.0 # fontScale
        color = (255, 0, 0) # purple color in BGR
        thickness = 6 # Line thickness of 2 px
        h, w, _ = img.shape
        for box in self.bbox:
            obj_score, _ = box[4], int(box[5])
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            org = (x1, y1)  # org
            cv2.putText(img, 'person: ' + str(round(obj_score, 3)), org, font,
                        fontScale, (0, 255, 255), thickness, cv2.LINE_AA)
        return img
