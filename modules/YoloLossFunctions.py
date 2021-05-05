import numpy as np
import tensorflow as tf
from . import Constants


# if bbox prediction predicts an object
def get_TP(pred, label):
    print()

# if bbox predicts an object that doesn't exist
def get_FP(pred, label):
    print()

# if an object is not detected by any bbox
def get_FN(pred, label):
    print()

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
        cells_per_axis = (Constants._MODEL_SIZE[0] // Constants.YOLO_STRIDES[s])
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

        for b in range(pred_stride.shape[0]):
            for cell in range(pred_stride.shape[1]):

                if mask_responsible[b][cell] == 1.0:

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
                    best_pred_i = np.argmax(iou)

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

        om = object_mask
        pd = np.asarray(pred_stride)
        pd = pd[:, :, :, 5:]
        sub = om - pd
        class_loss = np.square(sub)

        # combine bbox, obj, and class losses
        comb_loss = tf.concat([bbox_loss, obj_loss, class_loss], axis=-1)

        if total_loss is None:
            total_loss = comb_loss
        else:
            total_loss = tf.concat([total_loss, comb_loss], axis=1)

    return total_loss


def mask_responsible_cells(pred, label, cells_per_axis):
    model_dimension_size = Constants._MODEL_SIZE[0]
    cell_size = model_dimension_size / cells_per_axis

    mask_responsible = np.zeros(shape=[pred.shape[0], pred.shape[1]], dtype=np.float32)
    mask_responsible_for = np.zeros(shape=[pred.shape[0], pred.shape[1]], dtype=np.int32)

    for b in range(pred.shape[0]):  # for each batch
        for obj in range(label[b].shape[0]):  # for each object in the label

            label_x = label[b][obj][0]
            label_y = label[b][obj][1]

            cell_x = int(round(np.floor(label_x / cell_size)))
            cell_y = int(round(np.floor(label_y / cell_size)))

            cell = cell_y * cells_per_axis + cell_x

            mask_responsible[b][cell] = 1.0
            mask_responsible_for[b][cell] = obj

    return mask_responsible, mask_responsible_for


def mask_cells_objects(pred, label, cells_per_axis):
    # TODO: optimize; calculate prob's for each obj instead of for every cell
    model_dimension_size = Constants._MODEL_SIZE[0]
    cell_size = model_dimension_size / cells_per_axis

    obj_prob = np.zeros(shape=[pred.shape[0], pred.shape[1], 3, Constants.CLASSES])

    for b in range(pred.shape[0]):  # for each batch
        for cell in range(pred.shape[1]):  # for each cell
            for obj in range(label[b].shape[0]):

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
                cell_area = np.square(cell_size)

                overlap_ratio = overlap_area / cell_area

                if overlap_ratio > 1.0:
                    overlap_ratio = 1.0

                for anchor in range(3):
                    obj_prob[b][cell][anchor] = np.clip(
                        a=(obj_prob[b][cell][anchor] + overlap_ratio * label[b][obj][4:]),
                        a_max=1.0, a_min=0.0)

    return obj_prob
