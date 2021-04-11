import tensorflow as tf

CIFAR_MODEL_CHECKPOINT_PATH = "./ImgClassNN_0.1"


def model_cifar_01(training, classes=0, dropout_rate=0.0, trainingSpeed=0.001):
    inputs = tf.keras.Input(shape=(32, 32, 3))

    # conv and pool layers
    conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1, activation=tf.keras.activations.relu)(
        inputs)
    pool1 = tf.keras.layers.MaxPooling2D()(conv1)
    convDrop1 = tf.keras.layers.Dropout(rate=dropout_rate)(pool1)

    conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, activation=tf.keras.activations.relu)(
        convDrop1)
    pool2 = tf.keras.layers.MaxPooling2D()(conv2)
    convDrop2 = tf.keras.layers.Dropout(rate=dropout_rate / 2)(pool2)

    conv3 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1,
                                   activation=tf.keras.activations.relu)(convDrop2)
    pool3 = tf.keras.layers.MaxPooling2D()(conv3)
    convDrop3 = tf.keras.layers.Dropout(rate=dropout_rate / 3)(pool3)

    flatConv = tf.keras.layers.Flatten()(convDrop3)

    # fully connected layers
    dense1 = tf.keras.layers.Dense(units=1000, use_bias=True,
                                   activation=tf.keras.activations.relu)(flatConv)
    dropout1 = tf.keras.layers.Dropout(rate=dropout_rate)(dense1)
    dense2 = tf.keras.layers.Dense(units=256, use_bias=True,
                                   activation=tf.keras.activations.relu)(dropout1)
    dropout2 = tf.keras.layers.Dropout(rate=dropout_rate)(dense2)

    # output
    out = tf.keras.layers.Dense(units=classes, input_shape=(None, 256), use_bias=True,
                                activation=tf.keras.activations.softmax)(dropout2)

    # put model together
    model = tf.keras.Model(inputs=inputs, outputs=out)

    if training != True:
        model.compile()
        return model

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=trainingSpeed),
                  metrics=['accuracy'])

    model.summary()

    return model
