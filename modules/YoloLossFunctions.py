import numpy as np


def yolo_v3_loss_function(p_box_centers, p_box_shapes, p_confidence, p_classes, t_box_centers, t_box_shapes,
                          t_confidence, t_classes):
    box_center_loss = calc_box_center_loss(p_box_centers, t_box_centers)
    box_shape_loss = calc_box_shape_loss(p_box_shapes, t_box_shapes)
    confidence_loss = calc_confidence_loss(p_confidence, t_confidence)
    class_loss = calc_class_loss(p_classes, t_classes)

    return box_center_loss, box_shape_loss, confidence_loss, class_loss


def calc_box_center_loss(p_box_centers, t_box_centers):
    # for each batch
    for b in range(p_box_centers.shape[0]):

        n = 0
        # for each box found in batch
        for bx in range(p_box_centers.shape[1]):



def calc_box_shape_loss(p_box_shapes, t_box_shapes):


def calc_confidence_loss(p_confidence, t_confidence):


def calc_class_loss(p_classes, t_classes):
