"""
Constants.py

File for storing constants used by the Froogal Ads image network

Author: Byron Bingham
"""

import numpy as np

# FroogalAds contants
VERSION = "0.3"
LOAD_PRETRAINED_WEIGHTS = False
CLASS_LABEL_FILE = "./yolo_class_descriptions.csv"
ROLLING_AVG_LENGTH = 10000

# Training constants
# data should use channels last format
DATASET_PERCENTAGE = 100  # what percentage of dataset to use for training
CHECKPOINT_PATH = "./ImgClassNN_" + str(VERSION)  # where the checkpoints/weights are saved during training
SAVE_EVERY_N_BATCHES = 100  # save the weights every n batches
BATCH_SIZE = 4  # 4 was the biggest my PC could handle
EPOCH_SIZE = 100000  # number of batches
TRAINING_SPEED = 0.001  # learning rate. 0.001 seems to be standard starting rate for Yolo v3
CLASSES = 601  # number of classes for the data set/model
LOAD_WEIGHTS = True  # if true, loads weights from previous training runs
EPOCHS = 100
ZCA_EPSILON = 0.000001
YOLO_MAX_BBOX_PER_SCALE = 100
EVAL_BATCHES = 500  # how many batches used to calculate eval accuracy
ADAM_EPSILON = 0.001
GRAD_NORM = 1.0  # max value for gradients. Clipping gradients to prevent NaN issues
MAX_NAN_ERRORS = 4  # number of NaN errors before training is terminated
DO_EVAL_FIRST = False  # if true, the training script will do an eval step before beginning training
REDUCE_LEARNING_AFTER_EPOCHS = 10  # reduce the learning rate by "REDUCE_LEARNING_BY" every n epochs
REDUCE_LEARNING_BY = 2  # new learning rate = old rate / REDUCE_LEARNING_BY
IOU_LOSS_FACTOR = 1  # multiply IOU loss by this
CONF_LOSS_FACTOR = 1  # multiply confidence loss by this
CLASS_LOSS_FACTOR = 1  # multiply IOU loss by this

# YOLO constants
LEAKY_RELU = 0.1
ANCHORS = [[[10, 13], [16, 30], [33, 23]],
           [[30, 61], [62, 45], [59, 119]],
           [[116, 90], [156, 198], [373, 326]]]  # anchors based on data from COCO dataset
YOLO_STRIDES = [8, 16, 32]
MODEL_SIZE = (416, 416)
YOLO_IOU_LOSS_THRESH = 0.5

CONF_THRESHOLD = 0.3

ANCHORS_NORM = np.array(ANCHORS) / MODEL_SIZE[0]
