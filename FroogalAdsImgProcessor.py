import tensorflow as tf
from PIL import Image
import numpy as np
from modules import ImgClassModels
import pandas as pd
from modules import Constants
from modules import Yolov3
import matplotlib as matplot

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

        self.model = Yolov3.Create_Yolov3(input_size=Constants.MODEL_SIZE[0], channels=3, training=False,
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

        img_resize = img.resize(size=Constants.MODEL_SIZE)

        imgArr = tf.keras.preprocessing.image.img_to_array(img=img_resize, dtype='float32')
        imgArr = np.asarray([imgArr]) / 255.0
        print(imgArr.shape)
        with tf.device('/GPU:0'):
            prediction = self.model(imgArr)

        best_pred = []
        for s in range(3):
            tmp1 = np.array(prediction[s][..., 4])
            tmp2 = tf.reduce_mean(tmp1)
            tmp3 = tf.reduce_max(tmp1)
            tmp4 = float(tmp2) * 5 / (s + 1)
            best = np.argwhere(tmp1 > tmp4)
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
# imgProc.updateCSV(cls='cat', count=15)
# imgProc.updateCSV(cls='money', count=1)
