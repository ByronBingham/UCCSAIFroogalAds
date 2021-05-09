"""
FroogalAdsImgProcessor.py

This script is for running a process to classifiy objects in images and pass the data to the rest of the Froogal ads
service by editing a CSV file.

Calling "demoMode" will run a loop that will take in a file path to an image from the user and output an image with
bounding boxes on a copy of the given image.
Calling "classMode" will run a loop that will look for images in a directory. If images are found, they will be passed
in to the detection/classification network and the output classifications will be added to a CSV file

Author: Byron Bingham
"""

import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
from modules import Constants
from modules import Yolov3
import time
import os

CLASSES = 100
LABEL_NAME_PATH = "yolo_class_descriptions.csv"
CLASS_THRESHOLD = 0.9
CSV_PATH = "testcsv.csv"
IMAGES_DIR = "./images_in/"


class ImgProcessor:

    def loadLabelNames(self, labelFile):
        """
        Loads the names for each classification label. These are used to decode the one hot output of the model into
        strings that can be used to update the CSV file.
        :param labelFile:
        :return:
        """
        file = open(labelFile)
        while True:
            line = file.readline()
            line = line.lower()
            if len(line.strip()) == 0:
                break
            self.labelNames.append(line.strip())

    def __init__(self):
        tf.compat.v1.enable_control_flow_v2()
        print("eager or not: " + str(tf.executing_eagerly()))

        # create network
        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
        print(tf.config.experimental.list_physical_devices('GPU'))

        physical_devices = tf.config.experimental.list_physical_devices('GPU')

        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except Exception as e:
            print(e)

        self.labelNames = []
        self.loadLabelNames(Constants.CLASS_LABEL_FILE)

        if not Constants.LOAD_PRETRAINED_WEIGHTS:
            self.model = Yolov3.Create_Yolov3(input_size=Constants.MODEL_SIZE[0], channels=3, training=False,
                                              CLASSES=Constants.CLASSES)
            self.model.load_weights(Constants.CHECKPOINT_PATH)
        else:
            # this loads pre-trained weights trained on the COCO data set. These weights only have 80 classes
            self.model = Yolov3.Create_Yolov3(input_size=Constants.MODEL_SIZE[0], channels=3, training=False,
                                              CLASSES=80)
            Yolov3.load_yolo_weights(model=self.model, weights_file="yolov3.weights")

    def oneHotToLabelName(self, oneHotIn):
        """
        Decodes one-hot input into label strings for updating the CSV
        :param oneHotIn:
        :return:
        """
        max_index = np.argmax(oneHotIn)

        return self.labelNames[max_index]

    def classifyImage(self, imagePath):
        """
        Classifies objects in an image and updates the CSV with all classifications
        :param imagePath:
        :return:
        """
        img = Image.open(imagePath)

        img = img.resize(size=Constants.MODEL_SIZE)

        imgArr = tf.keras.preprocessing.image.img_to_array(img=img, dtype='float32') / 255.0
        imgArr = np.asarray([imgArr])
        print(imgArr.shape)
        with tf.device('/GPU:0'):
            prediction = self.model(imgArr)

        best_pred = []
        for s in range(3):
            pred_conf = np.array(prediction[s][..., 4])
            best = np.argwhere(pred_conf > Constants.CONF_THRESHOLD)
            if len(best) > 0:
                for b in best:
                    p = prediction[s][b[0]][b[1]][b[2]][b[3]]
                    best_pred.append(p)

        if len(best_pred) == 0:
            print("No objects detected")
            return

        for p in best_pred:
            class_one_hot = p[5:]
            name = self.oneHotToLabelName(class_one_hot)
            self.updateCSV(cls=name)

    def demoMode(self):
        """
        Runs a loop that asks for image files. Saves a copy of the given images with bounding boxes around objects.
        :return:
        """
        uInput = ""
        while True:
            uInput = input("Enter path to image or press \'q\' to exit:\n")
            if uInput.lower() == "q":
                break

            file = None

            # verify valid path
            try:
                file = open(uInput, "r")
                file.close()
            except Exception as e:
                print("ERROR: " + str(e) + "\nPlease enter a valid path")

            self.display_bboxes(uInput)

    def classMode(self):
        """
        Checks the specified directory for files (images) and classifies the objects in images whenever there are files
        in the directory.
        :return:
        """
        while True:
            for f in os.listdir(IMAGES_DIR):
                self.classifyImage(IMAGES_DIR + f)
                os.remove(IMAGES_DIR + f)
            time.sleep(3)

    def updateCSV(self, cls, count=1):
        """
        Adds the given class label to the CSV
        :param cls:
        :param count:
        :return:
        """
        print(cls)
        csv_dataframe = pd.read_csv(CSV_PATH, index_col=0)

        if cls in csv_dataframe.index:
            csv_dataframe.loc[cls, "count"] += count
        else:
            tmp = pd.DataFrame([[count]], columns=['count'], index=[cls])
            tmp.index.name = 'id'
            csv_dataframe = csv_dataframe.append(tmp)

        csv_dataframe.to_csv(path_or_buf=CSV_PATH)

    def display_bboxes(self, path):
        """
        For the specified image, saves a copy of the image that has bounding boxes around predicted objects
        :param path:
        :return:
        """
        img = Image.open(path)

        img_resize = img.resize(size=Constants.MODEL_SIZE)

        imgArr = tf.keras.preprocessing.image.img_to_array(img=img_resize, dtype='float32')
        imgArr = np.asarray([imgArr]) / 255.0
        print(imgArr.shape)
        with tf.device('/GPU:0'):
            prediction = self.model(imgArr)

        best_pred = []
        for s in range(3):
            tmp1 = np.array(prediction[s][..., 4])
            best = np.argwhere(tmp1 > Constants.CONF_THRESHOLD)
            if len(best) > 0:
                for b in best:
                    p = prediction[s][b[0]][b[1]][b[2]][b[3]]
                    best_pred.append(p)

        if len(best_pred) == 0:
            print("No objects detected. Try again")
            return

        best_pred = np.array(best_pred)

        bboxes = best_pred[..., 0:4] / Constants.MODEL_SIZE[0]

        x_min = bboxes[..., 0] - bboxes[..., 2] / 2.0
        y_min = bboxes[..., 1] - bboxes[..., 3] / 2.0
        x_max = bboxes[..., 0] + bboxes[..., 2] / 2.0
        y_max = bboxes[..., 1] + bboxes[..., 3] / 2.0

        x_min = tf.expand_dims(x_min, axis=-1)
        y_min = tf.expand_dims(y_min, axis=-1)
        x_max = tf.expand_dims(x_max, axis=-1)
        y_max = tf.expand_dims(y_max, axis=-1)

        tmp = bboxes
        bboxes = tf.concat([y_min, x_min, y_max, x_max], axis=-1)

        colors = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        img = tf.keras.preprocessing.image.img_to_array(img=img, dtype='float32')
        img_in = tf.expand_dims(input=img, axis=0)
        bboxes = tf.expand_dims(input=bboxes, axis=0)
        img = tf.image.draw_bounding_boxes(images=img_in, boxes=bboxes, colors=colors)[0]

        img = tf.cast(img, dtype=tf.uint8)
        img = np.array(img)
        img = Image.fromarray(img)
        img.save(fp=path + "_annotated.png")


imgProc = ImgProcessor()
imgProc.demoMode()
# imgProc.classMode()
# imgProc.updateCSV(cls='cat', count=15)
# imgProc.updateCSV(cls='money', count=1)
