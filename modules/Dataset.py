# ================================================================
#
#   File name   : dataset.py
#   Author      : PyLessons
#   Created date: 2020-07-31
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   Description : functions used to prepare dataset for custom training
#
# ================================================================
# TODO: transfer numpy to tensorflow operations

import os
import cv2
import random
import numpy as np
import tensorflow as tf
from . import Yolov3
from . import Constants
import tensorflow_datasets as tfds

"""
modified by Byron Bingham
"""


class YoloV3Dataset(object):
    # Dataset preprocess implementation
    def __init__(self, dataset_type):
        # self.input_sizes =
        self.batch_size = Constants.BATCH_SIZE

        if dataset_type is 'train':
            self.open_images_v4 = tfds.load("open_images_v4", shuffle_files=True,
                                            data_dir="x:/open_images_v4_dataset/",
                                            split='train[:' + str(Constants.DATASET_PERCENTAGE) + '%]')
        else:
            self.open_images_v4 = tfds.load("open_images_v4", shuffle_files=True,
                                            data_dir="x:/open_images_v4_dataset/",
                                            split='test[:' + str(Constants.DATASET_PERCENTAGE) + '%]')

        self.train_input_size = Constants._MODEL_SIZE[0]
        self.strides = np.array(Constants.YOLO_STRIDES)
        self.num_classes = Constants.CLASSES
        self.anchors = (np.array(Constants._ANCHORS).T / self.strides).T
        self.anchor_per_scale = 3
        self.max_bbox_per_scale = Constants.YOLO_MAX_BBOX_PER_SCALE

        self.batch_count = 0
        self.iter_start = 0

    def __iter__(self):
        return self

    def __next__(self):
        with tf.device('/cpu:0'):
            # self.train_input_size = random.choice([self.train_input_sizes])
            self.train_output_sizes = self.train_input_size // self.strides

            batch_image = np.zeros((self.batch_size, self.train_input_size, self.train_input_size, 3), dtype=np.float32)

            batch_label_sbbox = np.zeros((self.batch_size, self.train_output_sizes[0], self.train_output_sizes[0],
                                          self.anchor_per_scale, 5 + self.num_classes), dtype=np.float32)
            batch_label_mbbox = np.zeros((self.batch_size, self.train_output_sizes[1], self.train_output_sizes[1],
                                          self.anchor_per_scale, 5 + self.num_classes), dtype=np.float32)
            batch_label_lbbox = np.zeros((self.batch_size, self.train_output_sizes[2], self.train_output_sizes[2],
                                          self.anchor_per_scale, 5 + self.num_classes), dtype=np.float32)

            batch_sbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)
            batch_mbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)
            batch_lbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)

            ds_enum = self.open_images_v4.enumerate(start=self.iter_start)
            num = 0
            for element in ds_enum:
                if num > Constants.BATCH_SIZE - 1:
                    break
                self.iter_start += 1

                image = np.asarray(tf.cast(element[1]["image"], dtype=tf.float32))
                # original code was designed for coco data set, which its data is not normalized. Open images v4 is
                # normalized so we need to de-normalize it here
                bboxes = np.asarray(tf.cast(element[1]["bobjects"]["bbox"], dtype=tf.float32)) * Constants._MODEL_SIZE[
                    0]
                class_index = element[1]["bobjects"]["label"]
                class_index = np.expand_dims(class_index, axis=-1)
                bboxes = tf.concat([bboxes, class_index], axis=-1)

                label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.preprocess_true_boxes(bboxes)

                # batch_image[num, :, :, :] = tf.image.resize(image, size=Constants._MODEL_SIZE)
                batch_label_sbbox[num, :, :, :, :] = label_sbbox
                batch_label_mbbox[num, :, :, :, :] = label_mbbox
                batch_label_lbbox[num, :, :, :, :] = label_lbbox
                batch_sbboxes[num, :, :] = sbboxes
                batch_mbboxes[num, :, :] = mbboxes
                batch_lbboxes[num, :, :] = lbboxes
                num += 1

            self.batch_count += 1
            batch_smaller_target = batch_label_sbbox, batch_sbboxes
            batch_medium_target = batch_label_mbbox, batch_mbboxes
            batch_larger_target = batch_label_lbbox, batch_lbboxes

            return batch_image, (batch_smaller_target, batch_medium_target, batch_larger_target)

    def preprocess_true_boxes(self, bboxes):
        # TODO: fix for open images dataset
        label = [np.zeros((self.train_output_sizes[i], self.train_output_sizes[i], self.anchor_per_scale,
                           5 + self.num_classes)) for i in range(3)]
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(3)]
        bbox_count = np.zeros((3,))

        for bbox in bboxes:
            bbox_coor = bbox[:4]
            bbox_class_ind = int(bbox[4])

            onehot = np.zeros(self.num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0
            uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]

            iou = []
            exist_positive = False
            for i in range(3):
                anchors_xywh = np.zeros((self.anchor_per_scale, 4))
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                anchors_xywh[:, 2:4] = self.anchors[i]

                iou_scale = Yolov3.bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)

                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / self.anchor_per_scale)
                best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot

                bbox_ind = int(bbox_count[best_detect] % self.max_bbox_per_scale)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1
        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes
