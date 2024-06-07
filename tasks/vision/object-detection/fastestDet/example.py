#!/usr/bin/env python3
# Copyright 2024 NXP
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
import cv2
import argparse
from dperson import dperson

def parse_args():
    parser = argparse.ArgumentParser(
        description='person detection image test',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--model',
        dest='model',
        required=False,
        type=str,
        default='./dperson_shufflenetv2.tflite'
    )
    parser.add_argument(
        '--img',
        dest='img',
        required=False,
        type=str,
        default='./test.jpg'
    )
    parser.add_argument(
        '--video',
        dest='video',
        action='store_true',
        default=False
    )
    parser.add_argument(
        '--thresh',
        dest='thresh',
        required=False,
        type=float,
        default=0.6
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    cv2_window = "FastestDet NXP x86 Example"
    cv2.namedWindow(cv2_window, cv2.WINDOW_NORMAL) 
    cv2.resizeWindow(cv2_window, 640, 480)

    # instantiate detector model
    detector = dperson(args.model)

    # run demo on video
    if args.video:
        cap = cv2.VideoCapture(0)
        while True:
            _, img = cap.read()
            bboxes = detector.detect(img, args.thresh)

            # draw bounding boxes and show results
            img = detector.show(img)
            cv2.imshow(cv2_window, img)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    # run inference on example image
    else:
        img = cv2.imread(args.img)
        bboxes = detector.detect(img, args.thresh)

        # draw bounding boxes and show results
        detector.show(img)
        cv2.imshow(cv2_window, img)
        cv2.waitKey(0)
