# ================================================================
#
#   File name   : yolov3.py
#   Author      : PyLessons
#   Created date: 2020-06-04
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   Description : main yolov3 functions
#
# ================================================================
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, LeakyReLU, ZeroPadding2D, BatchNormalization, MaxPool2D
from tensorflow.keras.regularizers import l2
from . import Constants

STRIDES = np.array(Constants.YOLO_STRIDES)
ANCHORS = (np.array(Constants._ANCHORS).T / STRIDES).T


def bbox_iou(boxes1, boxes2):
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    return 1.0 * inter_area / union_area


def bbox_giou(boxes1, boxes2):
    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                        tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
    boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                        tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    # Calculate the iou value between the two bounding boxes
    iou = inter_area / union_area

    # Calculate the coordinates of the upper left corner and the lower right corner of the smallest closed convex surface
    enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)

    # Calculate the area of the smallest closed convex surface C
    enclose_area = enclose[..., 0] * enclose[..., 1]

    # Calculate the GIoU value according to the GioU formula
    giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

    return giou


def compute_loss(pred, label):  # bboxes,
    """
    No comments in original code. Commenting for myself to understand.

    :param pred: box predictions
    :param conv: one-hot prediction
    :param label: target label
    :param bboxes: bounding boxes
    :return:

    modified by Byron Bingham
    """

    box_loss = None
    objectiveness_loss = None
    prob_loss = None

    for i in range(3):

        NUM_CLASS = len(Constants.CLASSES)
        pred_shape = tf.shape(pred)
        batch_size = pred_shape[0]
        output_size = pred_shape[1]
        input_size = STRIDES[i] * output_size
        pred = tf.reshape(pred, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

        objectivness_score = pred[:, :, :, :, 4:5]  # class prediction confidence
        class_prob = pred[:, :, :, :, 5:]  # one-hot class predictions
        pred_xywh = pred[:, :, :, :, 0:4]  # predicted box dimensions and position

        label_xywh = label[:, :, :, :, 0:4]  # target box dimensions and position
        label_objectiveness = label[:, :, :, :, 4:5]  # how far from center: 1.0 is center, 0.0 is edge of bbox
        label_prob = label[:, :, :, :, 5:]  # target one-hot classes

        giou = tf.expand_dims(bbox_giou(pred_xywh, label_xywh), axis=-1)

        input_size = tf.cast(input_size, tf.float32)

        bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
        box_loss = label_objectiveness * bbox_loss_scale * (1 - giou)

        # iou = bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
        # Find the value of IoU with the real box The largest prediction box
        # max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

        # If the largest iou is less than the threshold, it is considered that the prediction box contains no objects, then the background box
        # respond_bgd = (1.0 - label_objectiveness) * tf.cast(max_iou < Constants.YOLO_IOU_LOSS_THRESH, tf.float32)

        conf_focal = tf.pow(label_objectiveness - objectivness_score, 2)

        # Calculate the loss of confidence
        # we hope that if the grid contains objects, then the network output prediction box has a confidence of 1 and 0 when there is no object.
        objectiveness_loss = conf_focal * (
                label_objectiveness * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_objectiveness,
                                                                              logits=class_prob)
            # +
            # respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_objectiveness, logits=class_prob)
        )

        prob_loss = label_objectiveness * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=class_prob)

        tmp_box_loss = tf.reduce_mean(
            tf.reduce_sum(box_loss, axis=[1, 2, 3, 4]))  # loss from dimensions/positions of boxes
        tmp_objectiveness_loss = tf.reduce_mean(tf.reduce_sum(objectiveness_loss, axis=[1, 2, 3, 4]))  # confidence loss
        tmp_prob_loss = tf.reduce_mean(
            tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))  # loss from one-hot class predictions

        if box_loss is None:
            box_loss = tmp_box_loss
            objectiveness_loss = tmp_objectiveness_loss
            prob_loss = tmp_prob_loss
        else:
            box_loss = tf.concat(box_loss, tmp_box_loss, axis=-2)
            objectiveness_loss = tf.concat(objectiveness_loss, tmp_objectiveness_loss, axis=-2)
            prob_loss = tf.concat(prob_loss, tmp_prob_loss, axis=-2)

    return box_loss, objectiveness_loss, prob_loss
