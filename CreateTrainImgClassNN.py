import tensorflow as tf
import numpy as np
import datetime as dt
import random
from modules import ImgClassModels
from matplotlib import pyplot
from modules import Constants
import tensorflow_datasets as tfds

open_images_v4_train = tfds.load("open_images_v4", shuffle_files=True, data_dir="x:/open_images_v4_dataset/",
                                 split='train[:' + str(Constants.DATASET_PERCENTAGE) + '%]')

open_images_v4_test = tfds.load("open_images_v4", shuffle_files=True, data_dir="x:/open_images_v4_dataset/",
                                split='test[:' + str(Constants.DATASET_PERCENTAGE) + '%]')


# test = tfds.load("open_images_v4", shuffle_files=True, data_dir="x:/open_images_v4_dataset/", split=['train[:1%]', 'test[:1%'])


class CreateTrainImgClassNN:

    def __init__(self):
        self.model = None
        self.train_acc_history = [(0.0, 0.0)]
        self.validation_acc_history = [(0.0, 0.0)]
        self.epochs = 0

        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

        self.train_start = 0
        self.test_start = 0

    def main(self):
        self.TF_Init()

        # create model
        (input_layer, output_layer) = ImgClassModels.getYoloModelLayers(model_size=Constants._MODEL_SIZE[0],
                                                                        n_classes=Constants.CLASSES, training=True)
        model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss=ImgClassModels.custom_yolo_cost,
                      optimizer=tf.keras.optimizers.Adam(learning_rate=Constants.TRAINING_SPEED),
                      metrics=['accuracy']) == model.summary()
        # load model checkpoint if exists
        # TODO: load darknet53/yolo weights
        """
        try:
            if Constants.LOAD_WEIGHTS:
                self.model.load_weights(Constants.CHECKPOINT_PATH)
        except Exception as e:
            print("Weights not loaded. Will create new weights")
            print(str(e))
        """

        # start training
        i = 0
        for i in range(0, Constants.TRAINING_STEPS):  # i < imagesToTrain:
            try:
                self.trainStep()
            except Exception as e:
                print("Training step failed. Skipping step")
                print(str(e) + "\n")
                continue

            print("Training step " + str(i) + " finished\n")

            if i % Constants.EVAL_FREQUENCY == Constants.EVAL_FREQUENCY - 1:
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

    def TF_Init(self):
        tf.compat.v1.enable_control_flow_v2()
        print("eager or not: " + str(tf.executing_eagerly()))

        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
        print(tf.config.experimental.list_physical_devices('GPU'))

        physical_devices = tf.config.experimental.list_physical_devices('GPU')

        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except Exception as e:
            print(e)

    def get_batch(self, training):
        print("Creating batch...")
        if training:
            ds = open_images_v4_train.enumerate(start=self.train_start)
        else:
            ds = open_images_v4_test.enumerate(start=self.test_start)

        x_batch = []
        y_batch = []

        elements_in_batch = 0
        for element in ds:
            if training:
                self.train_start += 1
            else:
                self.test_start += 1

            image = element[1]["image"]
            bbox = element[1]["bobjects"]["bbox"]
            label = element[1]["bobjects"]["label"]
            # TODO: change coordinates to xywh format
            x = image
            labels = []
            for l in label:
                labels.append(tf.one_hot(indices=l, depth=Constants.CLASSES, on_value=1.0, off_value=0.0))
            labels = np.asarray(labels)
            y = tf.concat([bbox, labels], axis=-1)

            x_batch.append(x)
            y_batch.append(y)
            elements_in_batch += 1
            if elements_in_batch >= Constants.BATCH_SIZE:
                break

        x_batch = np.asarray(x_batch) / 255
        y_batch = np.asarray(y_batch)

        print("Batch finished.")
        return x_batch, y_batch

    def evalStep(self):
        # trainData, targetData = self.createBatch()
        trainData = self.x_test
        targetData = self.y_test
        with tf.device('/GPU:0'):
            evalHistory = self.model.evaluate(x=trainData, y=targetData, verbose=2, batch_size=Constants.BATCH_SIZE)

        self.validation_acc_history.append((self.epochs, evalHistory[1]))

    def trainStep(self):

        # trainData, targetData = self.createBatch()
        time1 = dt.datetime.now()

        x_batch, y_batch = self.get_batch(training=True)

        epoch = self.model.fit(x=x_batch, y=y_batch, verbose=2, batch_size=Constants.BATCH_SIZE,
                               epochs=1)
        self.epochs = self.epochs + 1
        self.train_acc_history.append((self.epochs, epoch.history['accuracy'][0]))

        print("Time elapsed training: " + str(dt.datetime.now() - time1))

        self.model.save_weights(Constants.CHECKPOINT_PATH)
        print("Done with training step")

    def printResults(self):
        x, y = list(zip(*self.train_acc_history))
        pyplot.plot(x, y)

        x, y = list(zip(*self.validation_acc_history))
        pyplot.plot(x, y)

        pyplot.legend(labels=["Training", "Validation"])

        pyplot.ylabel("Accuracy")
        pyplot.xlabel("Epochs")
        pyplot.title("Training Accuracy vs Validation Accuracy\n" +
                     "B size = " + str(Constants.BATCH_SIZE) +
                     "; T speed = " + str(Constants.TRAINING_SPEED) +
                     "Dropout = " + str(Constants.DROPOUT_RATE) +
                     "D Aug = " + str(Constants.DATA_AUGMENTATION))

        pyplot.show()



if __name__ == '__main__':
    enhance = CreateTrainImgClassNN()
    enhance.main()
