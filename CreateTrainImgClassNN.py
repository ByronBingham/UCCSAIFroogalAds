import tensorflow as tf
import numpy as np
import datetime as dt
import random
from modules import ImgClassModels
from matplotlib import pyplot
from modules import Constants
import tensorflow_datasets as tfds
from modules import YoloLossFunctions as yloss
from modules import Yolov3
from modules import Dataset
from modules import YoloLossFunctions

"""
open_images_v4_train = tfds.load("open_images_v4", shuffle_files=True, data_dir="x:/open_images_v4_dataset/",
                                 split='train[:' + str(Constants.DATASET_PERCENTAGE) + '%]')

open_images_v4_test = tfds.load("open_images_v4", shuffle_files=True, data_dir="x:/open_images_v4_dataset/",
                                split='test[:' + str(Constants.DATASET_PERCENTAGE) + '%]')


# test = tfds.load("open_images_v4", shuffle_files=True, data_dir="x:/open_images_v4_dataset/", split=['train[:1%]', 'test[:1%'])
"""


class CreateTrainImgClassNN:

    def __init__(self):
        self.model = None
        self.train_acc_history = [(0.0, 0.0)]
        self.validation_acc_history = [(0.0, 0.0)]
        self.epochs = 0

        self.train_data = Dataset.YoloV3Dataset('train')
        self.test_data = Dataset.YoloV3Dataset('test')

    def main(self):
        self.TF_Init()

        # create model
        # (input_layer, output_layer) = ImgClassModels.getYoloModelLayers(model_size=Constants._MODEL_SIZE[0],
        #                                                                 n_classes=Constants.CLASSES, training=True)
        self.model = Yolov3.Create_Yolov3(input_size=Constants._MODEL_SIZE[0], channels=3, training=True,
                                          CLASSES=Constants.CLASSES)
        # self.model.compile(loss=ImgClassModels.custom_yolo_cost,
        #                   optimizer=tf.keras.optimizers.Adam(learning_rate=Constants.TRAINING_SPEED),
        #                   metrics=['accuracy']) ==
        self.model.summary()

        # load model checkpoint if exists
        # TODO: load darknet53/yolo weights

        try:
            if Constants.LOAD_WEIGHTS:
                self.model.load_weights(Constants.CHECKPOINT_PATH)
                print("Weights loaded")
        except Exception as e:
            print("Weights not loaded. Will create new weights")
            print(str(e))

        # start training
        i = 0
        for i in range(0, Constants.TRAINING_STEPS):  # i < imagesToTrain:
            self.trainStep()
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

    """
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

            image = np.asarray(tf.cast(element[1]["image"], dtype=tf.float32))
            bbox = np.asarray(tf.cast(element[1]["bobjects"]["bbox"], dtype=tf.float32))
            label = element[1]["bobjects"]["label"]
            # TODO: change coordinates to xywh format
            x = tf.image.resize(images=image, size=Constants._MODEL_SIZE)
            labels = []
            for l in label:
                labels.append(
                    np.asarray(tf.one_hot(indices=l, depth=Constants.CLASSES, on_value=1.0, off_value=0.0)).astype(
                        np.float32))
            labels = np.asarray(labels)
            y = tf.concat([bbox, labels], axis=-1)

            x = np.asarray(x)
            y = np.asarray(y)

            x_batch.append(x)
            y_batch.append(y)
            elements_in_batch += 1
            if elements_in_batch >= Constants.BATCH_SIZE:
                break

        x_batch = np.asarray(x_batch) / 255.0
        y_batch = np.asarray(y_batch)

        print("Batch finished.")
        return x_batch, y_batch
    """

    def evalStep(self):
        accuracy = 0.0

        batches = 0
        for batch in self.test_data:
            if batches > Constants.EVAL_BATCHES - 1:
                break

            x_batch, y_batch = batch

            model_output = self.model(x_batch, training=False)

            accuracy += self.eval_accuracy(model_output=model_output, label=y_batch)

        accuracy = accuracy / Constants.EVAL_BATCHES  # average accuracy over all batches

        print("Evaluation finished. Accuracy: " + str(accuracy))

        self.validation_acc_history.append((self.epochs, accuracy))

    def eval_accuracy(self, model_output, label):
        pred = model_output[1]

        tp = YoloLossFunctions.get_TP(pred=pred, label=label)
        fp = YoloLossFunctions.get_FP(pred=pred, label=label)
        fn = YoloLossFunctions.get_FN(pred=pred, label=label)

        prec = tp / (tp + fp)
        rec = tp / (tp + fn)

        accuracy = 2 * (prec * rec) / (prec + rec)
        return accuracy

    def trainStep(self):

        time1 = dt.datetime.now()
        for epoch in range(Constants.EPOCHS_PER_TRAINING_STEP):

            batches = 0
            for batch in self.train_data:
                if batches > Constants.EPOCH_SIZE - 1:
                    break

                giou_loss = conf_loss = prob_loss = 0
                model_out = None
                total_loss = None

                # batch = tf.where(tf.math.is_nan(batch), tf.zeros_like(batch), batch)
                x_batch, y_batch = batch

                optimizer = tf.keras.optimizers.Adam(learning_rate=Constants.TRAINING_SPEED,
                                                     epsilon=Constants.ADAM_EPSILON)

                with tf.GradientTape() as tape:
                    tape.watch(self.model.trainable_variables)
                    model_out = self.model(x_batch, training=True)

                    grid = 3
                    for i in range(grid):
                        conv, pred = model_out[i * 2], model_out[i * 2 + 1]
                        giou_loss_tmp, conf_loss_tmp, prob_loss_tmp = Yolov3.compute_loss(pred, conv, *y_batch[i],
                                                                                          i)
                        giou_loss += giou_loss_tmp
                        conf_loss += conf_loss_tmp
                        prob_loss += prob_loss_tmp

                    total_loss = giou_loss + conf_loss + prob_loss

                    if tf.math.is_nan(float(total_loss)):
                        print("NaN error. Resetting model weights to last checkpoint")
                        try:
                            self.model.load_weights(Constants.CHECKPOINT_PATH)
                            print("Weights loaded")
                            continue
                        except Exception as e:
                            print("Weights not loaded. Aborting training")
                            exit()
                    else:
                        gradients = tape.gradient(total_loss, self.model.trainable_variables)
                        gradients = [tf.clip_by_norm(g, Constants.GRAD_NORM) for g in gradients]
                        optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

                    print("Batch #" + str(batches) + "    Loss: " + str(float(total_loss)))

                batches += 1

                if (batches % Constants.SAVE_EVERY_N_BATCHES) >= Constants.SAVE_EVERY_N_BATCHES - 1:
                    print("Saving weights...")
                    self.model.save_weights(Constants.CHECKPOINT_PATH)

            print("Epoch " + str(epoch) + " finished. Evaluating model")
            # self.evalStep()

            self.epochs += 1

        print("Time elapsed training: " + str(dt.datetime.now() - time1))
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
