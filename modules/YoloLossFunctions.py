import numpy as np
import tensorflow as tf
from . import Constants

STRIDES = np.array(Constants.YOLO_STRIDES)


# ANCHORS = (np.array(Constants._ANCHORS).T / STRIDES).T


def bbox_iou(boxes1, boxes2):
    """
    from https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
    :param boxes1:
    :param boxes2:
    :return:
    """

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
    """
    from https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
    :param boxes1:
    :param boxes2:
    :return:
    """
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
    from https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
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
        label_objectiveness = label[:, :, :, :, 4:5]  # ?
        label_prob = label[:, :, :, :, 5:]  # target one-hot classes

        giou = tf.expand_dims(bbox_giou(pred_xywh, label_xywh), axis=-1)

        input_size = tf.cast(input_size, tf.float32)

        bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
        # box_loss = label_objectiveness * bbox_loss_scale * (1 - giou)

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


def compute_loss_custom(pred, label):
    """
    Computes the loss for the yolo v3 model

    :param pred: output of the yolo model. Format:
                [ ..., bbox xywh + objectiveness score + class probabilities ]
    :param label: target data. Format:
                [ batchs, objects, bbox xywh + class probabilities ]
                NOTE: class probabilities should be one-hot encoded
    :return: loss for training the yolo model. Format:
                [ ..., bbox loss + objectiveness loss + class loss ]

    author: Byron Bingham
    """

    # calculate the loss separately for each (of 3) detection layer (small, medium, large objects)
    total_loss = None
    previous_stride_cells = 0
    for s in range(len(Constants.YOLO_STRIDES)):
        cells_per_axis = (Constants._MODEL_SIZE[0] / Constants.YOLO_STRIDES[s])
        next_stride = previous_stride_cells + np.square(cells_per_axis)
        pred_stride = pred[:, previous_stride_cells:next_stride, ...]  # predictions for the current detection layer
        previous_stride_cells = next_stride

        model_dimension_size = Constants._MODEL_SIZE[0]
        stride_width = cells_per_axis / model_dimension_size

        # for each cell, find any centers that exist in the cell,
        # and any classes that exist in the cell and their iou with that cell
        mask_responsible, mask_responsible_for = mask_responsible_cells(pred=pred_stride, label=label,
                                                                        cells_per_axis=cells_per_axis)
        object_mask = mask_cells_objects(pred=pred_stride, label=label, cells_per_axis=cells_per_axis)

        # bbox loss
        # only penalize for bbox loss if cell is "responsible"
        # if a cell is responsible for predicting a target bbox
        #   loss = scaling factor * [(xt-xp)^2, (yt-yp)^2, (sqrt(wt) - sqrt(wp))^2, (sqrt(ht) - sqrt(hp))^2]
        # else
        #   loss = 0.0 (don't want to change anything if cell is not responsible for producing a bbox)

        bbox_loss = np.zeros(shape=[pred_stride.shape[0], pred_stride.shape[1], 3, 4])
        obj_loss = np.zeros(shape=[pred_stride.shape[0], pred_stride.shape[1], 3, 1])

        for b in range(len(pred_stride.shape[0])):
            for cell in range(len(pred_stride.shape[1])):

                if mask_responsible[b][cell] is 1.0:

                    iou = np.zeros(3)

                    # decide which anchor's bbox is closest to target
                    for anchor in range(3):
                        # sigmoid(prediction of x/y) + cell offset
                        pred_x = tf.math.sigmoid(pred_stride[b][cell][anchor][0]) + stride_width * (
                                cell % cells_per_axis)
                        pred_y = tf.math.sigmoid(pred_stride[b][cell][anchor][1]) + stride_width * (
                                cell % cells_per_axis)
                        # anchor w/h * e ^ (prediction of w/h)
                        pred_w = Constants.ANCHORS_NORM[s][anchor][0] * np.power(np.e, pred_stride[b][cell][anchor][2])
                        pred_h = Constants.ANCHORS_NORM[s][anchor][1] * np.power(np.e, pred_stride[b][cell][anchor][3])

                        pred_x_min = pred_x - pred_w / 2.0
                        pred_x_max = pred_x + pred_w / 2.0
                        pred_y_min = pred_y - pred_h / 2.0
                        pred_y_max = pred_y + pred_h / 2.0

                        label_x, label_y, label_w, label_h = label[b][mask_responsible_for[b][cell]][0:4]
                        label_x_min = label_x - label_w / 2.0
                        label_x_max = label_x + label_w / 2.0
                        label_y_min = label_y - label_h / 2.0
                        label_y_max = label_y + label_h / 2.0

                        x_overlap = min(pred_x_max, label_x_max) - max(pred_x_min, label_x_min)
                        y_overlap = min(pred_y_max, label_y_max) - max(pred_y_min, label_y_min)

                        intersection = x_overlap * y_overlap

                        total_area = pred_w * pred_h + label_w * label_h - intersection

                        iou[anchor] = intersection / total_area

                    # pick best prediction
                    best_pred_i = np.argmax(iou)[0]

                    # calculate loss for best
                    pred_x = tf.math.sigmoid(pred_stride[b][cell][best_pred_i][0]) + stride_width * (
                            cell % cells_per_axis)
                    pred_y = tf.math.sigmoid(pred_stride[b][cell][best_pred_i][1]) + stride_width * (
                            cell % cells_per_axis)
                    # anchor w/h * e ^ (prediction of w/h)
                    pred_w = Constants.ANCHORS_NORM[s][best_pred_i][0] * np.power(np.e,
                                                                                  pred_stride[b][cell][best_pred_i][2])
                    pred_h = Constants.ANCHORS_NORM[s][best_pred_i][1] * np.power(np.e,
                                                                                  pred_stride[b][cell][best_pred_i][3])
                    label_x, label_y, label_w, label_h = label[b][mask_responsible_for[b][cell]][0:4]

                    bbox_loss[b][cell][best_pred_i][0] = np.square(label_x - pred_x)
                    bbox_loss[b][cell][best_pred_i][1] = np.square(label_y - pred_y)
                    bbox_loss[b][cell][best_pred_i][2] = np.square(np.sqrt(label_w) - np.sqrt(pred_w))
                    bbox_loss[b][cell][best_pred_i][3] = np.square(np.sqrt(label_h) - np.sqrt(pred_h))

                    # objectiveness loss
                    # only penalize for obj. loss if cell is "responsible"
                    # if cell is responsible for obj.
                    #   loss = (ct-cp)^2
                    # else
                    #   loss = 0.0 (don't want to change anything if cell is not responsible for producing a bbox)
                    # only penalize best anchor prediction

                    obj_loss[b][cell][best_pred_i][0] = np.square(1.0 - pred_stride[b][cell][best_pred_i][4])

        # classification loss
        # penalize class loss if cell "contains" any part of an object
        # if there is an object(s) in the cell
        #   loss = (target - predicted)^2
        # else
        #   loss = 0.0 (don't want to change anything if there is nothing in the cell to detect/classify)

        class_loss = np.zeros(shape=[pred_stride.shape[0], pred_stride.shape[1], 3, Constants.CLASSES])

        for b in range(len(pred_stride.shape[0])):
            for cell in range(len(pred_stride.shape[1])):
                for anchor in range(3):
                    for cls in range(Constants.CLASSES):
                        class_loss[b][cell][anchor][cls] = np.square(
                            object_mask[b][cell][anchor][cls] - pred_stride[b][cell][anchor][5 + cls])

        # combine bbox, obj, and class losses
        comb_loss = tf.concat([bbox_loss, obj_loss, class_loss], axis=-1)

        if total_loss is None:
            total_loss = comb_loss
        else:
            total_loss = tf.concat([total_loss, comb_loss], axis=1)

    return total_loss


def mask_responsible_cells(pred, label, cells_per_axis):
    model_dimension_size = Constants._MODEL_SIZE[0]
    stride_width = tf.cast(cells_per_axis, dtype=tf.float32) / tf.cast(model_dimension_size,
                                                                       dtype=tf.float32)

    mask_responsible = np.ones(shape=[len(pred.shape[0]), len(pred.shape[1])], dtype=tf.float32)
    mask_responsible_for = np.zeros(shape=[len(pred.shape[0]), len(pred.shape[1])], dtype=tf.int32)

    for b in range(len(pred.shape[0])):  # for each batch
        for cell in range(len(pred.shape[1])):  # for each cell

            for obj in range(len(label.shape[-2])):

                x_min = stride_width * (cell % cells_per_axis)
                x_max = x_min + stride_width
                y_min = stride_width * (cell // cells_per_axis)
                y_max = y_min + stride_width

                label_x = label[b][obj][0]
                label_y = label[b][obj][1]

                # if the object's bbox center is within the cell, the cell is responsible
                if label_x >= x_min and label_x <= x_max and label_y >= y_min and label_y <= y_max:
                    mask_responsible[b][cell] = 1.0
                    mask_responsible_for[b][cell] = obj
                else:
                    mask_responsible[b][cell] = 0.0

    return mask_responsible, mask_responsible_for


def mask_cells_objects(pred, label, cells_per_axis):
    model_dimension_size = Constants._MODEL_SIZE[0]
    cell_size = tf.cast(cells_per_axis, dtype=tf.float32) / tf.cast(model_dimension_size,
                                                                    dtype=tf.float32)

    obj_prob = np.zeros(shape=[len(pred.shape[0]), len(pred.shape[1]), 3, Constants.CLASSES])

    for b in range(len(pred.shape[0])):  # for each batch
        for cell in range(len(pred.shape[1])):  # for each cell
            for obj in range(len(label.shape[-2])):

                x_min = cell_size * (cell % cells_per_axis)
                x_max = x_min + cell_size
                y_min = cell_size * (cell // cells_per_axis)
                y_max = y_min + cell_size

                label_x = label[b][obj][0]
                label_y = label[b][obj][1]
                label_w = label[b][obj][2]
                label_h = label[b][obj][3]

                label_x_min = label_x - label_w / 2
                label_x_max = label_x_min + label_w
                label_y_min = label_y - label_h / 2
                label_y_max = label_y_min + label_h

                if x_max < label_x_min:  # cell to the left of obj
                    continue
                if x_min > label_x_max:  # cell to the right of obj
                    continue
                if y_max < label_y_min:  # cell below obj
                    continue
                if y_min > label_y_max:  # cell above obj
                    continue

                # cell is not outside obj (cell and obj overlap)
                # calculate overlap

                x_overlap = min(x_max, label_x_max) - max(x_min, label_x_min)
                y_overlap = min(y_max, label_y_max) - max(y_min, label_y_min)

                overlap_area = x_overlap * y_overlap
                cell_area = np.sqaure(cell_size)

                overlap_ratio = overlap_area / cell_area

                if overlap_ratio > 1.0:
                    overlap_ratio = 1.0

                for anchor in range(3):
                    obj_prob[b][cell][anchor] = np.clip(a=(obj_prob[b][cell] + overlap_ratio * label[b][obj]),
                                                        a_max=1.0)

    return obj_prob
