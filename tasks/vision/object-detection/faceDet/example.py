# Copyright 2022-2023 NXP

import argparse

import cv2
import numpy as np
import tensorflow as tf


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp((-x)))


def nonMaxSuppress(boxLoc, score, maxBox=20, iouThresh=0.5):
    boxLoc = tf.cast(boxLoc, dtype=tf.float32)
    score = tf.cast(score, dtype=tf.float32)

    selected_indexes = tf.image.non_max_suppression(
        boxLoc, score, maxBox, iou_threshold=iouThresh
    )
    selected_boxes = tf.gather(boxLoc, selected_indexes)
    selected_score = tf.gather(score, selected_indexes)

    return selected_boxes, selected_score


def reciprocal_sigmoid(x):
    return -np.log(1 / x - 1)


def decode(netout, anchors, net_shape, image_shape, conf_thres=0.7, nms_score=0.45):
    grid_h, grid_w = netout.shape[:2]

    nb_box = 3
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))

    net_h, net_w = net_shape[1:3]
    image_h, image_w = image_shape[:2]

    netout[..., :2] = _sigmoid(netout[..., :2])
    netout[..., 4:] = _sigmoid(netout[..., 4:])
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * netout[..., 5:]

    if (float(net_w) / image_w) < (float(net_h) / image_h):
        new_w = net_w
        new_h = (image_h * net_w) / image_w
    else:
        new_h = net_w
        new_w = (image_w * net_h) / image_h

    x_offset, x_scale = (net_w - new_w) / 2.0 / net_w, float(net_w) / new_w
    y_offset, y_scale = (net_h - new_h) / 2.0 / net_h, float(net_h) / new_h

    cell_scores = netout[..., 4]
    cell_scores = np.expand_dims(cell_scores, axis=-1)

    col = [[j for j in range(grid_w)] for _ in range(grid_h)]
    row = [[i for _ in range(grid_w)] for i in range(grid_h)]

    col = np.array(col)
    row = np.array(row)

    col = np.reshape(col, newshape=(col.shape[0], col.shape[1], 1, 1))
    row = np.reshape(row, newshape=(row.shape[0], row.shape[1], 1, 1))

    # first 4 elements are x, y, w, and h
    x = np.expand_dims(netout[..., 0], axis=-1)
    y = np.expand_dims(netout[..., 1], axis=-1)
    w = np.expand_dims(netout[..., 2], axis=-1)
    h = np.expand_dims(netout[..., 3], axis=-1)

    x = (col + x) / grid_w  # center position, unit: image width
    y = (row + y) / grid_h  # center position, unit: image height

    anchors = np.expand_dims(anchors, axis=(0, 1))

    anchors_w = np.expand_dims(anchors[..., 0], axis=-1)
    anchors_h = np.expand_dims(anchors[..., 1], axis=-1)

    w = anchors_w * np.exp(w) / net_w  # unit: image width
    h = anchors_h * np.exp(h) / net_h  # unit: image height

    # last elements are class probabilities
    classes = netout[..., 5:]

    x = (x - x_offset) * x_scale
    y = (y - y_offset) * y_scale
    w *= x_scale
    h *= y_scale

    x2 = (x + w / 2) * image_w
    y2 = (y + h / 2) * image_h

    x2 = np.where(x2 >= image_w, image_w - 10, x2)
    y2 = np.where(y2 >= image_h, image_h - 10, y2)

    boxes = np.concatenate(
        [(x - w / 2) * image_w, (y - h / 2) * image_h, x2, y2, cell_scores, classes],
        axis=-1,
    )

    boxes_coord = boxes[..., :4]
    cell_scores = np.squeeze(cell_scores, axis=-1)
    mask = cell_scores > conf_thres

    boxes_over_thresh = tf.boolean_mask(boxes_coord, mask)
    scores_over_thresh = tf.boolean_mask(cell_scores, mask)

    if boxes_over_thresh.shape[0] == 0:
        return []

    selected_boxes, selected_score = nonMaxSuppress(
        boxes_over_thresh, scores_over_thresh, iouThresh=nms_score
    )
    selected_score = np.expand_dims(selected_score, axis=1)
    found_objects = np.concatenate((selected_boxes, selected_score), axis=1)

    return found_objects


def image_resize(ori_img, dst_shape):
    img_shape = ori_img.shape
    W, H = dst_shape
    scale = min(W / img_shape[1], H / img_shape[0])
    nw = int(img_shape[1] * scale)
    nh = int(img_shape[0] * scale)
    dx = (W - nw) // 2
    dy = (H - nh) // 2
    res_img = cv2.resize(ori_img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    new_img = np.ones((H, W, 3), np.uint8) * 128
    for i in range(nh):
        for j in range(nw):
            new_img[dy + i][dx + j] = res_img[i][j]

    return new_img


def model_detect(model, image, anchors):
    interpreter = tf.lite.Interpreter(model_path=str(model))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()
    shape = input_details["shape"]

    N, H, W, C = shape
    img_shape = image.shape
    res_img = image_resize(image, (W, H))
    img_data = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)

    img_data = img_data.reshape(shape)
    zero_point = input_details["quantization_parameters"]["zero_points"]
    scale = input_details["quantization_parameters"]["scales"]
    img_data = (img_data / 255 / scale + zero_point).astype("int8")

    interpreter.set_tensor(input_details["index"], img_data)
    interpreter.invoke()

    boxes = []
    for i in range(len(output_details)):
        anchor = anchors[i]
        out = interpreter.get_tensor(output_details[i]["index"])[0]
        zero_point = output_details[i]["quantization_parameters"]["zero_points"]
        scale = output_details[i]["quantization_parameters"]["scales"]
        out = ((out - zero_point) * scale).astype("float32")
        box = decode(out, anchor, shape, img_shape)
        if len(box) > 0:
            boxes.append(box)
    if len(boxes) == 0:
        return []
    return np.concatenate(boxes, axis=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-image", help="detect on image", default=r"./test.jpg", type=str
    )
    parser.add_argument(
        "-video", action="store_true", help="detect on CAM", default=False
    )
    args, unknown = parser.parse_known_args()
    anchors = [
        [[51, 64], [59, 82], [79, 100]],
        [[29, 51], [36, 43], [41, 54]],
        [[15, 21], [22, 29], [28, 36]],
    ]
    model = "./yolo_face_detect.tflite"
    if args.video:
        cap = cv2.VideoCapture(0)
        while True:
            ret, img = cap.read()
            if ret is False:
                break
            boxes = model_detect(model, img, anchors)
            # draw box
            if len(boxes) > 0:
                for box in boxes:
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    score = box[4]
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
                    cv2.putText(
                        img, "%.2f" % score, (x1, y1 - 5), 0, 0.7, (0, 255, 0), 2
                    )
            cv2.imshow("detector", img)
            # ESC to exit
            if cv2.waitKey(1) & 0xFF == 27:
                cv2.destroyAllWindows()
                break
    else:
        ori_img = cv2.imread(args.image)
        # detect in test images
        boxes = model_detect(model, ori_img, anchors)
        # draw box
        if len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                score = box[4]
                cv2.rectangle(ori_img, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(
                    ori_img, "%.2f" % score, (x1, y1 - 5), 0, 0.7, (0, 255, 0), 2
                )
        cv2.imwrite("result.jpg", ori_img)
