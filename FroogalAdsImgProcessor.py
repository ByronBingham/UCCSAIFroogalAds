import tensorflow as tf
from PIL import Image
import numpy as np
from modules import ImgClassModels
import pandas as pd

CLASSES = 100
MODEL_CHECKPOINT = "./model_cifar_0.2"
LABEL_NAME_PATH = "./cifar-100-fine_label_names.txt"
CLASS_THRESHOLD = 1.0
CSV_PATH = "./testcsv.csv"


class ImgProcessor:

    def loadLabelNames(self, labelFile):
        self.labelNames = []
        file = open(labelFile)
        while True:
            line = file.readline()
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

        self.model = ImgClassModels.model_cifar_01(classes=CLASSES, trainingSpeed=0.001,
                                                   dropout_rate=0.0, training=True)
        self.model.load_weights(MODEL_CHECKPOINT)
        self.loadLabelNames(LABEL_NAME_PATH)

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

            self.classifyImage(uInput)

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


imgProc = ImgProcessor()
# imgProc.demoMode()
# imgProc.updateCSV(cls='cat', count=15)
imgProc.updateCSV(cls='money', count=1)
