import numpy as np

# FroogalAds contants
VERSION = "0.3"
LOAD_PRETRAINED_WEIGHTS = False
CLASS_LABEL_FILE = "./yolo_class_descriptions.csv"

# Training constants
# data should use channels last format
DATASET_PERCENTAGE = 100
TRAINING_STEPS = 1
CHECKPOINT_PATH = "./ImgClassNN_" + str(VERSION)
SAVE_EVERY_N_BATCHES = 100
BATCH_SIZE = 4
EPOCH_SIZE = 10000  # number of batches
DROPOUT_RATE = 0.0
TRAINING_SPEED = 0.0001
CLASSES = 601
LOAD_WEIGHTS = True
EPOCHS = 10
ZCA_EPSILON = 0.000001
DATA_AUGMENTATION = True
YOLO_MAX_BBOX_PER_SCALE = 100
EVAL_BATCHES = 500
ADAM_EPSILON = 0.001
GRAD_NORM = 1.0
MAX_NAN_ERRORS = 4
DO_EVAL_FIRST = False
REDUCE_LEARNING_AFTER_EPOCHS = 2
REDUCE_LEARNING_BY = 2
IOU_LOSS_FACTOR = 50
CONF_LOSS_FACTOR = 1000
CLASS_LOSS_FACTOR = 1

# YOLO constants
CIFAR_MODEL_CHECKPOINT_PATH = "./ImgClassNN_0.1"
_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-05
_LEAKY_RELU = 0.1
_ANCHORS = [[[10, 13], [16, 30], [33, 23]],
            [[30, 61], [62, 45], [59, 119]],
            [[116, 90], [156, 198], [373, 326]]]  # anchors based on data from COCO dataset
YOLO_STRIDES = [8, 16, 32]
MODEL_SIZE = (416, 416)
TRAIN_DARKNET53 = True
YOLO_IOU_LOSS_THRESH = 0.5

CONF_THRESHOLD = 0.1

# ANCHORS_NORM = np.asarray(_ANCHORS / MODEL_SIZE[0])