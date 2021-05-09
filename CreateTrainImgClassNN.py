"""
CreateTrainImgClassNN.py

This script is for training the Froogal Ads image detection/classification network.

Author: Byron Bingham
"""

import tensorflow as tf
import numpy as np
import datetime as dt
from matplotlib import pyplot
from modules import Constants
from modules import Yolov3
from modules import Dataset
from modules import YoloLossFunctions


class CreateTrainImgClassNN:

    def __init__(self):
        self.model = None
        self.train_acc_history = [(0.0, 0.0)]
        self.validation_acc_history = [(0.0, 0.0)]
        self.epochs = 0

        self.train_data = Dataset.YoloV3Dataset('train')
        self.test_data = Dataset.YoloV3Dataset('test')

        self.learning_rate = Constants.TRAINING_SPEED

    def main(self):
        """
        Sets up Tensorflow parameters, creates a model and loads weights, and then starts training
        :return:
        """
        self.TF_Init()

        # create model
        self.model = Yolov3.Create_Yolov3(input_size=Constants.MODEL_SIZE[0], channels=3, training=True,
                                          CLASSES=Constants.CLASSES)

        self.model.summary()

        # load model checkpoint if exists
        try:
            if Constants.LOAD_WEIGHTS:
                self.model.load_weights(Constants.CHECKPOINT_PATH)
                print("Weights loaded")
        except Exception as e:
            print("Weights not loaded. Will create new weights")
            print(str(e))

        if Constants.DO_EVAL_FIRST:
            self.evalStep()

        # start training
        self.train()

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

    def evalStep(self):
        print("Evaluating...")
        bbox_accuracy, class_accuracy = (0.0, 0.0)

        batches = 0
        for batch in self.test_data:
            if batches > Constants.EVAL_BATCHES - 1:
                break

            x_batch, y_batch, num_of_objects = batch

            model_output = self.model(x_batch, training=False)

            bbox_accuracy_tmp, class_accuracy_tmp = self.eval_accuracy(pred=model_output, label=y_batch)
            bbox_accuracy += bbox_accuracy_tmp
            class_accuracy += class_accuracy_tmp

            print("Eval batch " + str(batches + 1) + "/" + str(Constants.EVAL_BATCHES) + " complete")
            batches += 1

        bbox_accuracy = bbox_accuracy / Constants.EVAL_BATCHES  # average accuracy over all batches
        class_accuracy = class_accuracy / Constants.EVAL_BATCHES

        print("Evaluation finished. Detection Accuracy: " + str(bbox_accuracy) + "    Classification Accuracy: " + str(
            class_accuracy))

        self.validation_acc_history.append((self.epochs, (bbox_accuracy + class_accuracy) / 2.0))

    def eval_accuracy(self, pred, label):

        tp_bbox = 0
        fp_bbox = 0
        fn_bbox = 0

        tp_class = 0
        fp_class = 0
        fn_class = 0

        for s in range(3):
            for b in range(len(pred[0])):
                decoded_pred = pred[s * 2 + 1][b]
                b_label = label[s]
                b_label = b_label[0][b]

                tmp_tp_bbox, tmp_fp_bbox, tmp_fn_bbox, tmp_tp_class, tmp_fp_class, tmp_fn_class = \
                    YoloLossFunctions.get_accuracy_metrics(pred=decoded_pred, label=b_label)

                tp_bbox += tmp_tp_bbox
                fp_bbox += tmp_fp_bbox
                fn_bbox += tmp_fn_bbox

                tp_class += tmp_tp_class
                fp_class += tmp_fp_class
                fn_class += tmp_fn_class

        bbox_accuracy, class_accuracy = (1.0, 1.0)

        try:
            prec = tp_bbox / (tp_bbox + fp_bbox)
            rec = tp_bbox / (tp_bbox + fn_bbox)

            bbox_accuracy = 2 * (prec * rec) / (prec + rec)

            prec = tp_class / (tp_class + fp_class)
            rec = tp_bbox / (tp_class + fn_class)

            class_accuracy = 2 * (prec * rec) / (prec + rec)
        except ZeroDivisionError:
            if np.all([tp_bbox, fp_bbox, fn_bbox] == 0):
                bbox_accuracy = 1.0
            else:
                bbox_accuracy = 0.0

            if np.all([tp_class, fp_class, fn_class] == 0):
                class_accuracy = 1.0
            else:
                class_accuracy = 0.0

        return bbox_accuracy, class_accuracy

    def train(self):
        """
        Trains the model.

        TODO: combine losses/gradients from smaller batches to effectively have larger batches
        :return:
        """
        time1 = dt.datetime.now()
        nan_errors = 0

        iou_loss_roll = np.array([])
        conf_loss_roll = np.array([])
        class_loss_roll = np.array([])

        for epoch in range(Constants.EPOCHS):

            batches = 0
            for batch in self.train_data:
                if batches > Constants.EPOCH_SIZE - 1:
                    break

                giou_loss = conf_loss = prob_loss = 0
                model_out = None
                total_loss = None

                # batch = tf.where(tf.math.is_nan(batch), tf.zeros_like(batch), batch)
                x_batch, y_batch, num_of_objects = batch

                optimizer = tf.keras.optimizers.Adam(learning_rate=Constants.TRAINING_SPEED,
                                                     epsilon=Constants.ADAM_EPSILON)

                with tf.GradientTape() as tape:
                    tape.watch(self.model.trainable_variables)
                    model_out = self.model(x_batch, training=True)

                    losses = []
                    for i in range(3):  # for each stride/grid size
                        conv, pred = model_out[i * 2], model_out[i * 2 + 1]
                        giou_loss_tmp, conf_loss_tmp, prob_loss_tmp = Yolov3.compute_loss(pred, conv, *y_batch[i],
                                                                                          i)

                        giou_loss_tmp = giou_loss_tmp * Constants.IOU_LOSS_FACTOR
                        conf_loss_tmp = conf_loss_tmp * Constants.CONF_LOSS_FACTOR
                        prob_loss_tmp = prob_loss_tmp * Constants.CLASS_LOSS_FACTOR

                        pred_loss = tf.concat(
                            [giou_loss_tmp, giou_loss_tmp, giou_loss_tmp, giou_loss_tmp, conf_loss_tmp, prob_loss_tmp],
                            axis=-1)
                        conv_loss = tf.reshape(pred_loss, shape=(
                            pred_loss.shape[0], pred_loss.shape[1], pred_loss.shape[2],
                            pred_loss.shape[3] * pred_loss.shape[4]))

                        losses.append(conv_loss)
                        losses.append(pred_loss)

                        giou_loss_tmp = tf.reduce_mean(tf.reduce_sum(giou_loss_tmp, axis=[1, 2, 3, 4]))
                        conf_loss_tmp = tf.reduce_mean(tf.reduce_sum(conf_loss_tmp, axis=[1, 2, 3, 4]))
                        prob_loss_tmp = tf.reduce_mean(tf.reduce_sum(prob_loss_tmp, axis=[1, 2, 3, 4]))

                        giou_loss += giou_loss_tmp
                        conf_loss += conf_loss_tmp
                        prob_loss += prob_loss_tmp

                    total_loss = giou_loss + conf_loss + prob_loss
                    iou_loss_roll = np.append(iou_loss_roll, giou_loss / num_of_objects)
                    if len(iou_loss_roll) > Constants.ROLLING_AVG_LENGTH:
                        np.delete(iou_loss_roll, 0)
                    conf_loss_roll = np.append(conf_loss_roll, conf_loss / num_of_objects)
                    if len(conf_loss_roll) > Constants.ROLLING_AVG_LENGTH:
                        np.delete(conf_loss_roll, 0)
                    class_loss_roll = np.append(class_loss_roll, prob_loss / num_of_objects)
                    if len(class_loss_roll) > Constants.ROLLING_AVG_LENGTH:
                        np.delete(class_loss_roll, 0)

                    if tf.math.is_nan(float(total_loss)):
                        if nan_errors > Constants.MAX_NAN_ERRORS:
                            print("Too many NaN errors. Aborting training...")
                            exit()
                        print("NaN error. Resetting model weights to last checkpoint")
                        try:
                            self.model.load_weights(Constants.CHECKPOINT_PATH)
                            print("Weights loaded")
                            continue
                        except Exception as e:
                            print("Weights not loaded. Aborting training")
                            exit()
                        nan_errors += 1
                    else:
                        gradients = tape.gradient(losses, self.model.trainable_variables)
                        gradients = [tf.clip_by_norm(g, Constants.GRAD_NORM) for g in gradients]
                        optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

                    print("Batch #" + str(batches) + "/" + str(Constants.EPOCH_SIZE) + " complete    Loss: " + str(
                        float(total_loss)))
                    print("IOU loss: " + str(float(giou_loss / num_of_objects)) + "    Conf loss: " + str(
                        float(conf_loss / num_of_objects)) + "    Class loss: " + str(
                        float(prob_loss / num_of_objects)))
                    print("Rolling average losses")
                    print("IOU loss: " + str(
                        np.sum(iou_loss_roll) / len(iou_loss_roll)) + "    Conf loss: " + str(
                        np.sum(conf_loss_roll) / len(conf_loss_roll)) + "    Class loss: " + str(
                        np.sum(class_loss_roll) / len(class_loss_roll)) + "\n")

                batches += 1

                if (batches % Constants.SAVE_EVERY_N_BATCHES) >= Constants.SAVE_EVERY_N_BATCHES - 1:
                    print("Saving weights...")
                    self.model.save_weights(Constants.CHECKPOINT_PATH)

            print("Epoch " + str(epoch) + " finished. Evaluating model")
            self.evalStep()

            if self.epochs % Constants.REDUCE_LEARNING_AFTER_EPOCHS >= Constants.REDUCE_LEARNING_AFTER_EPOCHS - 1:
                self.learning_rate = self.learning_rate / Constants.REDUCE_LEARNING_BY
                print("Updating learning rate. New learning rate: " + str(self.learning_rate))

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
                     "; T speed = " + str(Constants.TRAINING_SPEED))

        pyplot.show()


if __name__ == '__main__':
    enhance = CreateTrainImgClassNN()
    enhance.main()
