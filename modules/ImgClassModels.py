import tensorflow as tf
import numpy as np
from . import YoloLossFunctions
from . import Constants


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


def darknet_53_residual_block(input_layer, filters, training, strides=1):
    conv2d_1 = conv2d_fixed_padding(input_layer=input_layer, filters=filters, kernel_size=1, strides=strides)
    conv2d_1.trainable = Constants.TRAIN_DARKNET53

    batch_norm_1 = batch_norm(input_layer=conv2d_1, training=training)

    leaky_ReLU_1 = tf.keras.layers.LeakyReLU(alpha=Constants._LEAKY_RELU)(batch_norm_1)
    leaky_ReLU_1.trainable = Constants.TRAIN_DARKNET53

    conv2d_2 = conv2d_fixed_padding(input_layer=leaky_ReLU_1, filters=filters * 2, kernel_size=3, strides=strides)
    conv2d_2.trainable = Constants.TRAIN_DARKNET53

    batch_norm_2 = batch_norm(input_layer=conv2d_2, training=training)

    leaky_ReLU_2 = tf.keras.layers.LeakyReLU(alpha=Constants._LEAKY_RELU)(batch_norm_2)
    leaky_ReLU_2.trainable = Constants.TRAIN_DARKNET53

    return leaky_ReLU_2 + input_layer


def darknet_53_block(input_layer, training):
    # conv 32 3x3 1
    conv2d_1 = conv2d_fixed_padding(input_layer=input_layer, filters=32, kernel_size=3)
    conv2d_1.trainable = Constants.TRAIN_DARKNET53

    batch_norm_1 = batch_norm(input_layer=conv2d_1, training=training)

    leaky_ReLU_1 = tf.keras.layers.LeakyReLU(alpha=Constants._LEAKY_RELU)(batch_norm_1)
    leaky_ReLU_1.trainable = Constants.TRAIN_DARKNET53

    # conv 64 3x3 2
    conv2d_2 = conv2d_fixed_padding(input_layer=leaky_ReLU_1, filters=64, kernel_size=3, strides=2)
    conv2d_2.trainable = Constants.TRAIN_DARKNET53

    batch_norm_2 = batch_norm(input_layer=conv2d_2, training=training)

    leaky_ReLU_2 = tf.keras.layers.LeakyReLU(alpha=Constants._LEAKY_RELU)(batch_norm_2)
    leaky_ReLU_2.trainable = Constants.TRAIN_DARKNET53

    # residual 32/64 x1
    darknet53_res_layer_1 = darknet_53_residual_block(input_layer=leaky_ReLU_2, filters=32, training=training)

    # conv 128 3x3 2
    conv2d_3 = conv2d_fixed_padding(input_layer=darknet53_res_layer_1, filters=128, kernel_size=3, strides=2)
    conv2d_3.trainable = Constants.TRAIN_DARKNET53

    batch_norm_3 = batch_norm(input_layer=conv2d_3, training=training)

    leaky_ReLU_3 = tf.keras.layers.LeakyReLU(alpha=Constants._LEAKY_RELU)(batch_norm_3)
    leaky_ReLU_3.trainable = Constants.TRAIN_DARKNET53

    # residual 64/128 x2
    darknet53_res_layer_2 = darknet_53_residual_block(input_layer=leaky_ReLU_3, filters=64, training=training)

    darknet53_res_layer_3 = darknet_53_residual_block(input_layer=darknet53_res_layer_2, filters=64, training=training)
    # conv 256 3x3 2
    conv2d_4 = conv2d_fixed_padding(input_layer=darknet53_res_layer_3, filters=256, kernel_size=3, strides=2)
    conv2d_4.trainable = Constants.TRAIN_DARKNET53

    batch_norm_4 = batch_norm(input_layer=conv2d_4, training=training)

    leaky_ReLU_4 = tf.keras.layers.LeakyReLU(alpha=Constants._LEAKY_RELU)(batch_norm_4)
    leaky_ReLU_4.trainable = Constants.TRAIN_DARKNET53

    # residual 128/256 x8
    darknet53_res_layer_4 = darknet_53_residual_block(input_layer=leaky_ReLU_4, filters=128, training=training)
    darknet53_res_layer_5 = darknet_53_residual_block(input_layer=darknet53_res_layer_4, filters=128, training=training)
    darknet53_res_layer_6 = darknet_53_residual_block(input_layer=darknet53_res_layer_5, filters=128, training=training)
    darknet53_res_layer_7 = darknet_53_residual_block(input_layer=darknet53_res_layer_6, filters=128, training=training)
    darknet53_res_layer_8 = darknet_53_residual_block(input_layer=darknet53_res_layer_7, filters=128, training=training)
    darknet53_res_layer_9 = darknet_53_residual_block(input_layer=darknet53_res_layer_8, filters=128, training=training)
    darknet53_res_layer_10 = darknet_53_residual_block(input_layer=darknet53_res_layer_9, filters=128,
                                                       training=training)
    darknet53_res_layer_11 = darknet_53_residual_block(input_layer=darknet53_res_layer_10, filters=128,
                                                       training=training)

    route1 = darknet53_res_layer_11

    # conv 512 3x3 2
    conv2d_5 = conv2d_fixed_padding(input_layer=darknet53_res_layer_11, filters=512, kernel_size=3, strides=2)
    conv2d_5.trainable = Constants.TRAIN_DARKNET53

    batch_norm_5 = batch_norm(input_layer=conv2d_5, training=training)

    leaky_ReLU_5 = tf.keras.layers.LeakyReLU(alpha=Constants._LEAKY_RELU)(batch_norm_5)
    leaky_ReLU_5.trainable = Constants.TRAIN_DARKNET53

    # residual 256/512 x8
    darknet53_res_layer_12 = darknet_53_residual_block(input_layer=leaky_ReLU_5, filters=256, training=training)
    darknet53_res_layer_13 = darknet_53_residual_block(input_layer=darknet53_res_layer_12, filters=256,
                                                       training=training)
    darknet53_res_layer_14 = darknet_53_residual_block(input_layer=darknet53_res_layer_13, filters=256,
                                                       training=training)
    darknet53_res_layer_15 = darknet_53_residual_block(input_layer=darknet53_res_layer_14, filters=256,
                                                       training=training)
    darknet53_res_layer_16 = darknet_53_residual_block(input_layer=darknet53_res_layer_15, filters=256,
                                                       training=training)
    darknet53_res_layer_17 = darknet_53_residual_block(input_layer=darknet53_res_layer_16, filters=256,
                                                       training=training)
    darknet53_res_layer_18 = darknet_53_residual_block(input_layer=darknet53_res_layer_17, filters=256,
                                                       training=training)
    darknet53_res_layer_19 = darknet_53_residual_block(input_layer=darknet53_res_layer_18, filters=256,
                                                       training=training)

    route2 = darknet53_res_layer_19

    # conv 1024 3x3 2
    conv2d_6 = conv2d_fixed_padding(input_layer=darknet53_res_layer_19, filters=1024, kernel_size=3, strides=2)
    conv2d_6.trainable = Constants.TRAIN_DARKNET53

    batch_norm_6 = batch_norm(input_layer=conv2d_6, training=training)

    leaky_ReLU_6 = tf.keras.layers.LeakyReLU(alpha=Constants._LEAKY_RELU)(batch_norm_6)
    leaky_ReLU_6.trainable = Constants.TRAIN_DARKNET53

    # residual 512/1024 x4
    darknet53_res_layer_20 = darknet_53_residual_block(input_layer=leaky_ReLU_6, filters=512, training=training)
    darknet53_res_layer_21 = darknet_53_residual_block(input_layer=darknet53_res_layer_20, filters=512,
                                                       training=training)
    darknet53_res_layer_22 = darknet_53_residual_block(input_layer=darknet53_res_layer_21, filters=512,
                                                       training=training)
    darknet53_res_layer_23 = darknet_53_residual_block(input_layer=darknet53_res_layer_22, filters=512,
                                                       training=training)

    return route1, route2, darknet53_res_layer_23


def yolo_conv_block(input_layer, filters, training):
    """ Creates convolution operations layer used after Darknet.
        modified from https://www.kaggle.com/aruchomu/yolo-v3-object-detection-in-tensorflow
    """
    conv2d_1 = conv2d_fixed_padding(input_layer=input_layer, filters=filters, kernel_size=1)
    batch_norm_1 = batch_norm(input_layer=conv2d_1, training=training)
    leakyReLU_1 = tf.keras.layers.LeakyReLU(alpha=Constants._LEAKY_RELU)(batch_norm_1)

    conv2d_2 = conv2d_fixed_padding(input_layer=leakyReLU_1, filters=2 * filters, kernel_size=3)
    batch_norm_2 = batch_norm(input_layer=conv2d_2, training=training)
    leakyReLU_2 = tf.keras.layers.LeakyReLU(alpha=Constants._LEAKY_RELU)(batch_norm_2)

    conv2d_3 = conv2d_fixed_padding(input_layer=leakyReLU_2, filters=filters, kernel_size=1)
    batch_norm_3 = batch_norm(conv2d_3, training=training)
    leakyReLU_3 = tf.keras.layers.LeakyReLU(alpha=Constants._LEAKY_RELU)(batch_norm_3)

    conv2d_4 = conv2d_fixed_padding(input_layer=leakyReLU_3, filters=2 * filters, kernel_size=3)
    batch_norm_4 = batch_norm(input_layer=conv2d_4, training=training)
    leakyReLU_4 = tf.keras.layers.LeakyReLU(alpha=Constants._LEAKY_RELU)(batch_norm_4)

    conv2d_5 = conv2d_fixed_padding(input_layer=leakyReLU_4, filters=filters, kernel_size=1)
    batch_norm_5 = batch_norm(input_layer=conv2d_5, training=training)
    leakyReLU_5 = tf.keras.layers.LeakyReLU(alpha=Constants._LEAKY_RELU)(batch_norm_5)

    route = leakyReLU_5

    conv2d_6 = conv2d_fixed_padding(input_layer=leakyReLU_5, filters=2 * filters, kernel_size=3)
    batch_norm_6 = batch_norm(input_layer=conv2d_6, training=training)
    leakyReLU_6 = tf.keras.layers.LeakyReLU(alpha=Constants._LEAKY_RELU)(batch_norm_6)

    return route, leakyReLU_6


def yolo_detection_layer(input_layer, n_classes, anchors, img_size):
    """Creates Yolo final detection layer.

        Detects boxes with respect to anchors.

        Args:
            input_layer: Tensor input.
            n_classes: Number of labels.
            anchors: A list of anchor sizes.
            img_size: The input size of the model.

        Returns:
            Tensor output.

        modified from https://www.kaggle.com/aruchomu/yolo-v3-object-detection-in-tensorflow
        """

    n_anchors = len(anchors)

    conv2d_1 = tf.keras.layers.Conv2D(filters=n_anchors * (5 + n_classes),
                                      kernel_size=1, strides=1, use_bias=True)(input_layer)

    shape = input_layer.get_shape().as_list()
    grid_shape = shape[1:3]
    conv2d_1 = tf.reshape(conv2d_1, [-1, n_anchors * grid_shape[0] * grid_shape[1],
                                     5 + n_classes])

    strides = (img_size[0] // grid_shape[0], img_size[1] // grid_shape[1])

    box_centers, box_shapes, confidence, classes = tf.split(conv2d_1, [2, 2, 1, n_classes], axis=-1)

    x = tf.range(grid_shape[0], dtype=tf.float32)
    y = tf.range(grid_shape[1], dtype=tf.float32)
    x_offset, y_offset = tf.meshgrid(x, y)
    x_offset = tf.reshape(x_offset, (-1, 1))
    y_offset = tf.reshape(y_offset, (-1, 1))
    x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
    x_y_offset = tf.tile(x_y_offset, [1, n_anchors])
    x_y_offset = tf.reshape(x_y_offset, [1, -1, 2])
    box_centers = tf.nn.sigmoid(box_centers)
    box_centers = (box_centers + x_y_offset) * strides

    anchors = tf.tile(anchors, [grid_shape[0] * grid_shape[1], 1])
    box_shapes = tf.exp(box_shapes) * tf.cast(x=anchors, dtype=tf.float32)

    confidence = tf.nn.sigmoid(confidence)

    classes = tf.nn.sigmoid(classes)

    inputs = tf.concat([box_centers, box_shapes,
                        confidence, classes], axis=-1)

    return inputs


def upsample_layer(input_layer, out_shape):
    """
        Upsamples to `out_shape` using nearest neighbor interpolation.
        modified from https://www.kaggle.com/aruchomu/yolo-v3-object-detection-in-tensorflow
    """

    new_height = out_shape[2]
    new_width = out_shape[1]

    out = tf.compat.v1.image.resize_nearest_neighbor(input_layer, (new_height, new_width))

    return out


def batch_norm(input_layer, training):
    """Performs a batch normalization using a standard set of parameters.
    modified from https://www.kaggle.com/aruchomu/yolo-v3-object-detection-in-tensorflow
    """
    tmp = tf.keras.layers.BatchNormalization(axis=3,
                                             momentum=Constants._BATCH_NORM_DECAY,
                                             epsilon=Constants._BATCH_NORM_EPSILON,
                                             scale=True, trainable=training)(input_layer)
    return tmp


def fixed_padding(input_layer, kernel_size):
    """ResNet implementation of fixed padding.

    Pads the input along the spatial dimensions independently of input size.

    Args:
        input_layer: Tensor input to be padded.
        kernel_size: The kernel to be used in the conv2d or max_pool2d.
    Returns:
        A tensor with the same format as the input.

    modified from https://www.kaggle.com/aruchomu/yolo-v3-object-detection-in-tensorflow
    """
    pad_total = kernel_size - 1

    padded_inputs = tf.keras.layers.ZeroPadding2D(padding=pad_total)(input_layer)
    return padded_inputs


def conv2d_fixed_padding(input_layer, filters, kernel_size, strides=1):
    """
        Strided 2-D convolution with explicit padding.
        modified from https://www.kaggle.com/aruchomu/yolo-v3-object-detection-in-tensorflow
    """
    if strides > 1:
        out = fixed_padding(input_layer=input_layer, kernel_size=kernel_size)
    else:
        out = input_layer

    return tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                  strides=strides, padding=('SAME' if strides == 1 else 'VALID'), use_bias=False)(out)


def build_boxes(inputs):
    """
        Computes top left and bottom right points of the boxes.
        modified from https://www.kaggle.com/aruchomu/yolo-v3-object-detection-in-tensorflow
    """
    center_x, center_y, width, height, confidence, classes = \
        tf.split(inputs, [1, 1, 1, 1, 1, -1], axis=-1)

    top_left_x = center_x - width / 2
    top_left_y = center_y - height / 2
    bottom_right_x = center_x + width / 2
    bottom_right_y = center_y + height / 2

    boxes = tf.concat([top_left_x, top_left_y,
                       bottom_right_x, bottom_right_y], axis=-1)

    return boxes, confidence, classes


def non_max_suppression(inputs, n_classes, max_output_size, iou_threshold,
                        confidence_threshold):
    """
    Performs non-max suppression separately for each class.

    Args:
        inputs: Tensor input.
        n_classes: Number of classes.
        max_output_size: Max number of boxes to be selected for each class.
        iou_threshold: Threshold for the IOU.
        confidence_threshold: Threshold for the confidence score.
    Returns:
        A list containing class-to-boxes dictionaries
            for each sample in the batch.

    modified from https://www.kaggle.com/aruchomu/yolo-v3-object-detection-in-tensorflow
    """
    batch = tf.unstack(inputs)
    boxes_dicts = []
    for boxes in batch:
        boxes = tf.boolean_mask(boxes, boxes[:, 4] > confidence_threshold)
        classes = tf.argmax(boxes[:, 5:], axis=-1)
        classes = tf.expand_dims(tf.cast(x=classes, dtype=tf.float32), axis=-1)
        boxes = tf.concat([boxes[:, :5], classes], axis=-1)

        boxes_dict = dict()
        for cls in range(n_classes):
            mask = tf.equal(boxes[:, 5], cls)
            mask_shape = mask.get_shape()
            if mask_shape.ndims != 0:
                class_boxes = tf.boolean_mask(boxes, mask)
                boxes_coords, boxes_conf_scores, _ = tf.split(class_boxes,
                                                              [4, 1, -1],
                                                              axis=-1)
                boxes_conf_scores = tf.reshape(boxes_conf_scores, [-1])
                indices = tf.image.non_max_suppression(boxes_coords,
                                                       boxes_conf_scores,
                                                       max_output_size,
                                                       iou_threshold)
                class_boxes = tf.gather(class_boxes, indices)
                boxes_dict[cls] = class_boxes[:, :5]

        boxes_dicts.append(boxes_dict)

    return boxes_dicts


def getYoloModelLayers(model_size, n_classes, training):
    """
        returns an input layer and output layers to be used with keras Model class
    """
    input_layer = tf.keras.layers.Input(shape=(model_size, model_size, 3), dtype=tf.float32)

    (dn_route1, dn_route2, darknet_1) = darknet_53_block(input_layer=input_layer, training=training)
    yolo_route_1, conv_1 = yolo_conv_block(input_layer=darknet_1, filters=512, training=training)
    detect_1 = yolo_detection_layer(input_layer=conv_1, n_classes=n_classes,
                                    anchors=Constants._ANCHORS[6:9],
                                    img_size=(model_size, model_size))
    pad_1 = conv2d_fixed_padding(input_layer=yolo_route_1, filters=256, kernel_size=1)
    batch_norm_1 = batch_norm(input_layer=pad_1, training=training)
    leakyReLU_1 = tf.keras.layers.LeakyReLU(alpha=Constants._LEAKY_RELU)(batch_norm_1)
    upsample_size = dn_route2.get_shape().as_list()
    upsample_1 = upsample_layer(input_layer=leakyReLU_1, out_shape=upsample_size)
    axis = 3
    concat_1 = tf.concat([upsample_1, dn_route2], axis=axis)

    ####################################################################################################################

    yolo_route_2, conv_2 = yolo_conv_block(input_layer=concat_1, filters=256, training=training)
    detect_2 = yolo_detection_layer(input_layer=conv_2, n_classes=n_classes, anchors=Constants._ANCHORS[3:6],
                                    img_size=(model_size, model_size))
    pad_2 = conv2d_fixed_padding(input_layer=yolo_route_2, filters=128, kernel_size=1)
    batch_norm_2 = batch_norm(input_layer=pad_2, training=training)
    leakyReLU_2 = tf.keras.layers.LeakyReLU(alpha=Constants._LEAKY_RELU)(batch_norm_2)
    upsample_size = dn_route1.get_shape().as_list()
    upsample_2 = upsample_layer(input_layer=leakyReLU_2, out_shape=upsample_size)
    concat_2 = tf.concat([upsample_2, dn_route1], axis=axis)

    ####################################################################################################################

    yolo_route_3, conv_3 = yolo_conv_block(
        input_layer=concat_2, filters=128, training=training)
    detect_3 = yolo_detection_layer(input_layer=conv_3, n_classes=n_classes, anchors=Constants._ANCHORS[0:3],
                                    img_size=(model_size, model_size))

    out = tf.concat([detect_1, detect_2, detect_3], axis=1)

    return input_layer, out


def format_output_for_yolo_loss(data):
    box_centers, box_shapes, confidence, classes = tf.split(data, [2, 2, 1, Constants.CLASSES], axis=-1)
    return box_centers, box_shapes, confidence, classes


def custom_yolo_cost(y_true, y_pred):
    # pass boxes to yolo loss function
    box_loss, objectiveness_loss, prob_loss = YoloLossFunctions.compute_loss(pred=y_pred, label=y_true)

    # return losses
    out = tf.concat([box_loss, objectiveness_loss, prob_loss], axis=-1)
    return out
