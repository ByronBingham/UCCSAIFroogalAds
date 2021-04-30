import numpy as np

# FroogalAds contants
VERSION = 0.2

# Training constants
# data should use channels last format
DATASET_PERCENTAGE = 5
TRAINING_STEPS = 100
CHECKPOINT_PATH = "./ImgClassNN_" + str(VERSION)
BATCH_SIZE = 32
EPOCH_SIZE = (10000, 10000, 10000, 10000)
DROPOUT_RATE = 0.0
TRAINING_SPEED = 0.001
CLASSES = 601
LOAD_WEIGHTS = False
EVAL_FREQUENCY = 1
EPOCHS_PER_TRAINING_STEP = (1, 1, 1, 1)
ZCA_EPSILON = 0.000001
DATA_AUGMENTATION = True


# YOLO constants
CIFAR_MODEL_CHECKPOINT_PATH = "./ImgClassNN_0.1"
_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-05
_LEAKY_RELU = 0.1
_ANCHORS = [(10, 13), (16, 30), (33, 23),
            (30, 61), (62, 45), (59, 119),
            (116, 90), (156, 198), (373, 326)]  # anchors based on data from COCO dataset
YOLO_STRIDES = [8, 16, 32]
_MODEL_SIZE = (416, 416)
TRAIN_DARKNET53 = False
YOLO_IOU_LOSS_THRESH = 0.5

ANCHORS_NORM = np.asarray([[(_ANCHORS[0][0] / _MODEL_SIZE[0], _ANCHORS[0][1] / _MODEL_SIZE[0]),
                           (_ANCHORS[1][0] / _MODEL_SIZE[0], _ANCHORS[1][1] / _MODEL_SIZE[0]),
                           (_ANCHORS[2][0] / _MODEL_SIZE[0], _ANCHORS[2][1] / _MODEL_SIZE[0])],
                           [(_ANCHORS[3][0] / _MODEL_SIZE[0], _ANCHORS[3][1] / _MODEL_SIZE[0]),
                            (_ANCHORS[4][0] / _MODEL_SIZE[0], _ANCHORS[4][1] / _MODEL_SIZE[0]),
                            (_ANCHORS[5][0] / _MODEL_SIZE[0], _ANCHORS[5][1] / _MODEL_SIZE[0])],
                           [(_ANCHORS[6][0] / _MODEL_SIZE[0], _ANCHORS[6][1] / _MODEL_SIZE[0]),
                            (_ANCHORS[7][0] / _MODEL_SIZE[0], _ANCHORS[7][1] / _MODEL_SIZE[0]),
                            (_ANCHORS[8][0] / _MODEL_SIZE[0], _ANCHORS[8][1] / _MODEL_SIZE[0])]])
