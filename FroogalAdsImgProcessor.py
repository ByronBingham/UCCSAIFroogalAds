import tensorflow as tf
from PIL import Image
import numpy as np
from modules import ImgClassModels
import pandas as pd
from modules import Constants
from modules import Yolov3

CLASSES = 100
LABEL_NAME_PATH = ""
CLASS_THRESHOLD = 1.0
CSV_PATH = "./testcsv.csv"


class ImgProcessor:
    """
    def loadLabelNames(self, labelFile):

        file = open(labelFile)
        while True:
            line = file.readline()
            if len(line.strip()) == 0:
                break
            self.labelNames.append(line.strip())
    """

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

        self.model = Yolov3.Create_Yolov3(input_size=Constants.MODEL_SIZE[0], channels=3, training=True,
                                          CLASSES=Constants.CLASSES)
        self.model.load_weights(Constants.CHECKPOINT_PATH)
        # self.loadLabelNames(LABEL_NAME_PATH)

        self.labelNames = []

    def oneHotToLabelName(self, oneHotIn):
        i = 0
        for i in range(0, len(oneHotIn[0])):
            tmp = oneHotIn[0][i]
            if tmp >= CLASS_THRESHOLD:
                break

        return self.labelNames[i]

    def classifyImage(self, imagePath):
        img = Image.open(imagePath)

        img.resize(size=(32, 32))

        imgArr = tf.keras.preprocessing.image.img_to_array(img=img, dtype='float32')
        imgArr = np.asarray([imgArr])
        print(imgArr.shape)
        with tf.device('/GPU:0'):
            prediction = self.model.predict(x=imgArr)

        print("Prediction:\n")
        print(str(prediction))
        print("Label: " + self.oneHotToLabelName(prediction))

    def demoMode(self):
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

    def updateCSV(self, cls, count=1):
        csv_dataframe = pd.read_csv(CSV_PATH, index_col=0)
        print(csv_dataframe)
        if cls in csv_dataframe.index:
            csv_dataframe.loc[cls, "count"] += count
        else:
            tmp = pd.DataFrame([[count]], columns=['count'], index=[cls])
            tmp.index.name = 'id'
            print(tmp)
            csv_dataframe = csv_dataframe.append(tmp)

        print(csv_dataframe)
        csv_dataframe.to_csv(path_or_buf=CSV_PATH)

    def display_bboxes(self, path):
        img = Image.open(path)

        img = img.resize(size=Constants.MODEL_SIZE)

        imgArr = tf.keras.preprocessing.image.img_to_array(img=img, dtype='float32')
        imgArr = np.asarray([imgArr])
        print(imgArr.shape)
        with tf.device('/GPU:0'):
            prediction = self.model(imgArr)

        best_pred = []
        for s in [1, 3, 5]:
            tmp1 = prediction[s][0, ..., 4]
            tmp2 = tf.reduce_mean(tmp1)
            best = np.argwhere(tmp1 > tmp2)
            if len(best) > 0:
                best_pred.append(best)

        if len(best_pred) is 0:
            print("No objects detected. Try again")
            return

        best_pred = np.array(best_pred)


imgProc = ImgProcessor()
imgProc.demoMode()
# imgProc.updateCSV(cls='cat', count=15)
# imgProc.updateCSV(cls='money', count=1)
