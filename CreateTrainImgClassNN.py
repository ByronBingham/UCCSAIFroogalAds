import tensorflow as tf
import numpy as np
import datetime as dt
import random
from modules import ImgClassModels
from matplotlib import pyplot

TRAINING_STEPS = 100
CHECKPOINT_PATH = ImgClassModels.CIFAR_MODEL_CHECKPOINT_PATH
BATCH_SIZE = 32
EPOCH_SIZE = (10000, 10000, 10000, 10000)
DROPOUT_RATE = 0.0
TRAINING_SPEED = 0.001
CLASSES = 100
LOAD_WEIGHTS = False
EVAL_FREQUENCY = 1
EPOCHS_PER_TRAINING_STEP = (1, 1, 1, 1)
ZCA_EPSILON = 0.000001
DATA_AUGMENTATION = True


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


class CreateTrainImgClassNN:

    def oneHot(self, integer):
        arr = np.zeros(shape=CLASSES, dtype='float32')
        arr[integer] = 1.0
        return arr

    def oneHotArr(self, inArr):
        arr = np.zeros(shape=(len(inArr), CLASSES), dtype='float32')
        for i in range(len(inArr)):
            arr[i] = self.oneHot(inArr[i])

        return arr

    def __init__(self):
        self.model = None
        self.train_acc_history = [(0.0, 0.0)]
        self.validation_acc_history = [(0.0, 0.0)]
        self.epochs = 0
        # self.dataset = unpickle("./cifar-100-python/train")
        # self.testData = unpickle("./cifar-100-python/test")

        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.cifar100.load_data()

        # reshape to be [samples][width][height][channels]
        self.x_train = self.x_train.reshape((self.x_train.shape[0], 32, 32, 3))
        self.x_test = self.x_test.reshape((self.x_test.shape[0], 32, 32, 3))
        # convert from int to float
        self.x_train = self.x_train.astype('float32')
        self.x_test = self.x_test.astype('float32')

        self.y_train = self.oneHotArr(self.y_train)
        self.y_test = self.oneHotArr(self.y_test)

        # of don't divide by 255, values are out of expected range for float format
        # e.g. highest float32 value for color is supposed to be 1.0; when converting from int, the value does not
        # change.
        # int (0 - 255) -> float (0.0 - 1.0)
        self.x_train = self.x_train / 255
        self.x_test = self.x_test / 255

        self.transDataGen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=5, horizontal_flip=True,
                                                                            vertical_flip=True)

        self.normDataGen = tf.keras.preprocessing.image.ImageDataGenerator(featurewise_center=True,
                                                                           featurewise_std_normalization=True)

        self.zcaDataGen = tf.keras.preprocessing.image.ImageDataGenerator(zca_whitening=True,
                                                                          zca_epsilon=ZCA_EPSILON)

        self.comboDataGen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=5, horizontal_flip=True,
                                                                            vertical_flip=True, fill_mode='nearest',
                                                                            zca_whitening=True, zca_epsilon=ZCA_EPSILON)

        self.transDataGen.fit(self.x_train)
        self.normDataGen.fit(self.x_train)
        self.zcaDataGen.fit(self.x_train)
        self.comboDataGen.fit(self.x_train)

        self.augIterators = (
            self.transDataGen.flow(x=self.x_train, y=self.y_train,
                                   batch_size=EPOCH_SIZE[0], shuffle=True),
            self.normDataGen.flow(x=self.x_train, y=self.y_train,
                                  batch_size=EPOCH_SIZE[1], shuffle=True),
            self.zcaDataGen.flow(x=self.x_train, y=self.y_train,
                                 batch_size=EPOCH_SIZE[2], shuffle=True),
            self.comboDataGen.flow(x=self.x_train, y=self.y_train,
                                   batch_size=EPOCH_SIZE[3], shuffle=True)
        )

    def main(self):

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

        self.model = ImgClassModels.model_cifar_01(classes=CLASSES, trainingSpeed=TRAINING_SPEED,
                                                   dropout_rate=DROPOUT_RATE, training=True)

        # load model checkpoint if exists
        try:
            if LOAD_WEIGHTS:
                self.model.load_weights(CHECKPOINT_PATH)
        except Exception as e:
            print("Weights not loaded. Will create new weights")
            print(str(e))

        # start training
        i = 0
        for i in range(0, TRAINING_STEPS):  # i < imagesToTrain:
            try:
                self.trainStep()
            except Exception as e:
                print("Training step failed. Skipping step")
                print(str(e) + "\n")
                continue

            print("Training step " + str(i) + " finished\n")

            if i % EVAL_FREQUENCY == EVAL_FREQUENCY - 1:
                try:
                    print("Evaluating model...")
                    self.evalStep()
                    print("\n")

                except Exception as e:
                    print("Training step failed. Skipping step")
                    print(str(e) + "\n")
                    continue

            i += 1

        self.printResults()

    def evalStep(self):
        # trainData, targetData = self.createBatch()
        trainData = self.x_test
        targetData = self.y_test
        with tf.device('/GPU:0'):
            evalHistory = self.model.evaluate(x=trainData, y=targetData, verbose=2, batch_size=BATCH_SIZE)

        self.validation_acc_history.append((self.epochs, evalHistory[1]))

    def trainStep(self):

        # trainData, targetData = self.createBatch()
        time1 = dt.datetime.now()

        epoch = self.model.fit(x=self.x_train, y=self.y_train, verbose=2, batch_size=BATCH_SIZE,
                               epochs=1)
        self.epochs = self.epochs + 1
        self.train_acc_history.append((self.epochs, epoch.history['accuracy'][0]))

        if DATA_AUGMENTATION:
            for it in range(0, len(self.augIterators)):
                print("Creating aug data for training step...")
                tmp = self.augIterators[it]
                trainBatch, targetBatch = tmp.next()
                print("Time elapsed creating data: " + str(dt.datetime.now() - time1))

                # debug
                # for i in range(0, 9):
                #    pyplot.subplot(330 + 1 + i)
                #    pyplot.imshow(trainBatch[i])
                # show the plot
                # pyplot.show()
                # debug

                with tf.device('/GPU:0'):
                    epoch = self.model.fit(x=trainBatch, y=targetBatch, verbose=2, batch_size=BATCH_SIZE,
                                           epochs=EPOCHS_PER_TRAINING_STEP[it])

                self.train_acc_history.append((self.epochs, epoch.history['accuracy'][0]))
                self.epochs = self.epochs + 1

        print("Time elapsed training: " + str(dt.datetime.now() - time1))

        self.model.save_weights(CHECKPOINT_PATH)
        print("Done with training step")

    def createBatch(self):
        train = np.zeros(shape=(BATCH_SIZE, 32, 32, 3))
        label = np.zeros(shape=(BATCH_SIZE, CLASSES))

        for i in range(BATCH_SIZE):
            rand = random.randint(0, len(self.dataset[b'data']) - 1)
            train[i] = self.dataset[b'data'][rand]
            label[i] = self.dataset[b'fine_labels'][rand]

        return train, label

    def printResults(self):
        x, y = list(zip(*self.train_acc_history))
        pyplot.plot(x, y)

        x, y = list(zip(*self.validation_acc_history))
        pyplot.plot(x, y)

        pyplot.legend(labels=["Training", "Validation"])

        pyplot.ylabel("Accuracy")
        pyplot.xlabel("Epochs")
        pyplot.title("Training Accuracy vs Validation Accuracy\n" +
                     "B size = " + str(BATCH_SIZE) +
                     "; T speed = " + str(TRAINING_SPEED) +
                     "Dropout = " + str(DROPOUT_RATE) +
                     "D Aug = " + str(DATA_AUGMENTATION))

        pyplot.show()


if __name__ == '__main__':
    enhance = CreateTrainImgClassNN()
    enhance.main()
