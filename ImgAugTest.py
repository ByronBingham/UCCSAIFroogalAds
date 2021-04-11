# import tensorflow as tf
import numpy as np
from matplotlib import pyplot

x, y = list(zip(*[(1, 1), (2, 3), (4, 5)]))
pyplot.plot(x, y)

x, y = list(zip(*[(5, 1), (6, 3), (7, 5)]))
pyplot.plot(x, y)
pyplot.legend(labels=["Training", "Validation"])

pyplot.ylabel("Accuracy")
pyplot.xlabel("Epochs")
pyplot.title("Training Accuracy vs Validation Accuracy\n" +
             "B size = 32; T speed = 0.001; etc...")

pyplot.show()

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar100.load_data()

# reshape to be [samples][width][height][channels]
X_train = X_train.reshape((X_train.shape[0], 32, 32, 3))
X_test = X_test.reshape((X_test.shape[0], 32, 32, 3))
# convert from int to float
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

datagen = tf.keras.preprocessing.image.ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)

datagen.fit(X_train)

# configure batch size and retrieve one batch of images
for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=50000, shuffle=True):
    # create a grid of 3x3 images
    for i in range(0, 9):
        pyplot.subplot(330 + 1 + i)
        pyplot.imshow(X_batch[i])
    # show the plot
    pyplot.show()
    break
