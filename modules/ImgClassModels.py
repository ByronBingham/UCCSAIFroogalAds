import tensorflow as tf

CIFAR_MODEL_CHECKPOINT_PATH = "./ImgClassNN_0.1"
_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-05
_LEAKY_RELU = 0.1
_ANCHORS = [(10, 13), (16, 30), (33, 23),
            (30, 61), (62, 45), (59, 119),
            (116, 90), (156, 198), (373, 326)]  # anchors based on data from COCO dataset
_MODEL_SIZE = (416, 416)


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


def darknet_53_residual_block(input_layer, filters, data_format, training, strides=1):
    conv2d_1 = conv2d_fixed_padding(input_layer=input_layer, filters=filters, kernel_size=1, data_format=data_format,
                                    strides=strides)

    batch_norm_1 = batch_norm(input_layer=conv2d_1, training=training, data_format=data_format)

    leaky_ReLU_1 = tf.keras.layers.LeakyReLU(alpha=_LEAKY_RELU)(batch_norm_1)

    conv2d_2 = conv2d_fixed_padding(input_layer=leaky_ReLU_1, filters=filters, kernel_size=3, data_format=data_format,
                                    strides=strides)

    batch_norm_2 = batch_norm(input_layer=conv2d_2, training=training, data_format=data_format)

    leaky_ReLU_2 = tf.keras.layers.LeakyReLU(alpha=_LEAKY_RELU)(batch_norm_2)

    return leaky_ReLU_2 + input_layer


def darknet_53_block(input_layer, data_format, training):
    conv2d_1 = conv2d_fixed_padding(input_layer=input_layer, filters=32, kernel_size=3, data_format=data_format)

    batch_norm_1 = batch_norm(input_layer=conv2d_1, training=training, data_format=data_format)

    leaky_ReLU_1 = tf.keras.layers.LeakyReLU(alpha=_LEAKY_RELU)(batch_norm_1)

    conv2d_2 = conv2d_fixed_padding(input_layer=leaky_ReLU_1, filters=32, kernel_size=3, data_format=data_format,
                                    strides=2)

    batch_norm_2 = batch_norm(input_layer=conv2d_2, training=training, data_format=data_format)

    leaky_ReLU_2 = tf.keras.layers.LeakyReLU(alpha=_LEAKY_RELU)(batch_norm_2)

    darknet53_res_layer_1 = darknet_53_residual_block(input_layer=leaky_ReLU_2, filters=32, training=training,
                                                      data_format=data_format)

    conv2d_3 = conv2d_fixed_padding(input_layer=darknet53_res_layer_1, filters=128, kernel_size=3,
                                    data_format=data_format,
                                    strides=2)

    batch_norm_3 = batch_norm(input_layer=conv2d_3, training=training, data_format=data_format)

    leaky_ReLU_3 = tf.keras.layers.LeakyReLU(alpha=_LEAKY_RELU)(batch_norm_3)

    darknet53_res_layer_2 = darknet_53_residual_block(input_layer=leaky_ReLU_3, filters=64, training=training,
                                                      data_format=data_format)

    darknet53_res_layer_3 = darknet_53_residual_block(input_layer=darknet53_res_layer_2, filters=64, training=training,
                                                      data_format=data_format)

    conv2d_4 = conv2d_fixed_padding(input_layer=darknet53_res_layer_3, filters=256, kernel_size=3,
                                    data_format=data_format,
                                    strides=2)

    batch_norm_4 = batch_norm(input_layer=conv2d_4, training=training, data_format=data_format)

    leaky_ReLU_4 = tf.keras.layers.LeakyReLU(alpha=_LEAKY_RELU)(batch_norm_4)

    darknet53_res_layer_4 = darknet_53_residual_block(input_layer=leaky_ReLU_4, filters=128, training=training,
                                                      data_format=data_format)
    darknet53_res_layer_5 = darknet_53_residual_block(input_layer=darknet53_res_layer_4, filters=128, training=training,
                                                      data_format=data_format)
    darknet53_res_layer_6 = darknet_53_residual_block(input_layer=darknet53_res_layer_5, filters=128, training=training,
                                                      data_format=data_format)
    darknet53_res_layer_7 = darknet_53_residual_block(input_layer=darknet53_res_layer_6, filters=128, training=training,
                                                      data_format=data_format)
    darknet53_res_layer_8 = darknet_53_residual_block(input_layer=darknet53_res_layer_7, filters=128, training=training,
                                                      data_format=data_format)
    darknet53_res_layer_9 = darknet_53_residual_block(input_layer=darknet53_res_layer_8, filters=128, training=training,
                                                      data_format=data_format)
    darknet53_res_layer_10 = darknet_53_residual_block(input_layer=darknet53_res_layer_9, filters=128,
                                                       training=training,
                                                       data_format=data_format)
    darknet53_res_layer_11 = darknet_53_residual_block(input_layer=darknet53_res_layer_10, filters=128,
                                                       training=training,
                                                       data_format=data_format)

    route1 = darknet53_res_layer_11

    conv2d_5 = conv2d_fixed_padding(input_layer=darknet53_res_layer_11, filters=512, kernel_size=3,
                                    data_format=data_format,
                                    strides=2)

    batch_norm_5 = batch_norm(input_layer=conv2d_5, training=training, data_format=data_format)

    leaky_ReLU_5 = tf.keras.layers.LeakyReLU(alpha=_LEAKY_RELU)(batch_norm_5)

    darknet53_res_layer_12 = darknet_53_residual_block(input_layer=leaky_ReLU_5, filters=256, training=training,
                                                       data_format=data_format)
    darknet53_res_layer_13 = darknet_53_residual_block(input_layer=darknet53_res_layer_12, filters=256,
                                                       training=training,
                                                       data_format=data_format)
    darknet53_res_layer_14 = darknet_53_residual_block(input_layer=darknet53_res_layer_13, filters=256,
                                                       training=training,
                                                       data_format=data_format)
    darknet53_res_layer_15 = darknet_53_residual_block(input_layer=darknet53_res_layer_14, filters=256,
                                                       training=training,
                                                       data_format=data_format)
    darknet53_res_layer_16 = darknet_53_residual_block(input_layer=darknet53_res_layer_15, filters=256,
                                                       training=training,
                                                       data_format=data_format)
    darknet53_res_layer_17 = darknet_53_residual_block(input_layer=darknet53_res_layer_16, filters=256,
                                                       training=training,
                                                       data_format=data_format)
    darknet53_res_layer_18 = darknet_53_residual_block(input_layer=darknet53_res_layer_17, filters=256,
                                                       training=training,
                                                       data_format=data_format)
    darknet53_res_layer_19 = darknet_53_residual_block(input_layer=darknet53_res_layer_18, filters=256,
                                                       training=training,
                                                       data_format=data_format)

    route2 = darknet53_res_layer_19

    conv2d_6 = conv2d_fixed_padding(input_layer=darknet53_res_layer_19, filters=1024, kernel_size=3,
                                    data_format=data_format, strides=2)

    batch_norm_6 = batch_norm(input_layer=conv2d_6, training=training, data_format=data_format)

    leaky_ReLU_6 = tf.keras.layers.LeakyReLU(alpha=_LEAKY_RELU)(batch_norm_6)

    darknet53_res_layer_20 = darknet_53_residual_block(input_layer=leaky_ReLU_6, filters=512, training=training,
                                                       data_format=data_format)
    darknet53_res_layer_21 = darknet_53_residual_block(input_layer=darknet53_res_layer_20, filters=512,
                                                       training=training,
                                                       data_format=data_format)
    darknet53_res_layer_22 = darknet_53_residual_block(input_layer=darknet53_res_layer_21, filters=512,
                                                       training=training,
                                                       data_format=data_format)
    darknet53_res_layer_23 = darknet_53_residual_block(input_layer=darknet53_res_layer_22, filters=512,
                                                       training=training,
                                                       data_format=data_format)

    return route1, route2, darknet53_res_layer_23


def yolo_conv_block(input_layer, filters, data_format, training):
    """ Creates convolution operations layer used after Darknet.
        modified from https://www.kaggle.com/aruchomu/yolo-v3-object-detection-in-tensorflow
    """
    conv2d_1 = conv2d_fixed_padding(input_layer=input_layer, filters=filters, kernel_size=1,
                                    data_format=data_format)
    batch_norm_1 = batch_norm(input_layer=conv2d_1, training=training, data_format=data_format)
    leakyReLU_1 = tf.nn.leaky_relu(alpha=_LEAKY_RELU)(batch_norm_1)

    conv2d_2 = conv2d_fixed_padding(input_layer=leakyReLU_1, filters=2 * filters, kernel_size=3,
                                    data_format=data_format)
    batch_norm_2 = batch_norm(input_layer=conv2d_2, training=training, data_format=data_format)
    leakyReLU_2 = tf.nn.leaky_relu(alpha=_LEAKY_RELU)(batch_norm_2)

    conv2d_3 = conv2d_fixed_padding(input_layer=leakyReLU_2, filters=filters, kernel_size=1,
                                    data_format=data_format)
    batch_norm_3 = batch_norm(conv2d_3, training=training, data_format=data_format)
    leakyReLU_3 = tf.nn.leaky_relu(alpha=_LEAKY_RELU)(batch_norm_3)

    conv2d_4 = conv2d_fixed_padding(input_layer=leakyReLU_3, filters=2 * filters, kernel_size=3,
                                    data_format=data_format)
    batch_norm_4 = batch_norm(input_layer=conv2d_4, training=training, data_format=data_format)
    leakyReLU_4 = tf.nn.leaky_relu(alpha=_LEAKY_RELU)(batch_norm_4)

    conv2d_5 = conv2d_fixed_padding(input_layer=leakyReLU_4, filters=filters, kernel_size=1,
                                    data_format=data_format)
    batch_norm_5 = batch_norm(input_layer=conv2d_5, training=training, data_format=data_format)
    leakyReLU_5 = tf.nn.leaky_relu(alpha=_LEAKY_RELU)(batch_norm_5)

    route = leakyReLU_5

    conv2d_6 = conv2d_fixed_padding(input_layer=leakyReLU_5, filters=2 * filters, kernel_size=3,
                                    data_format=data_format)
    batch_norm_6 = batch_norm(input_layer=conv2d_6, training=training, data_format=data_format)
    leakyReLU_6 = tf.nn.leaky_relu(alpha=_LEAKY_RELU)(batch_norm_6)

    return route, leakyReLU_6


def yolo_detection_layer(input_layer, n_classes, anchors, img_size, data_format):
    """Creates Yolo final detection layer.

        Detects boxes with respect to anchors.

        Args:
            input_layer: Tensor input.
            n_classes: Number of labels.
            anchors: A list of anchor sizes.
            img_size: The input size of the model.
            data_format: The input format.

        Returns:
            Tensor output.

        modified from https://www.kaggle.com/aruchomu/yolo-v3-object-detection-in-tensorflow
        """

    n_anchors = len(anchors)

    conv2d_1 = tf.keras.layers.Conv2D(filters=n_anchors * (5 + n_classes),
                                      kernel_size=1, strides=1, use_bias=True,
                                      data_format=data_format)(input_layer)

    shape = input_layer.get_shape().as_list()
    grid_shape = shape[2:4] if data_format == 'channels_first' else shape[1:3]
    if data_format == 'channels_first':
        conv2d_1 = tf.transpose(conv2d_1, [0, 2, 3, 1])
    conv2d_1 = tf.reshape(conv2d_1, [-1, n_anchors * grid_shape[0] * grid_shape[1],
                                     5 + n_classes])

    strides = (img_size[0] // grid_shape[0], img_size[1] // grid_shape[1])

    box_centers, box_shapes, confidence, classes = \
        tf.split(conv2d_1, [2, 2, 1, n_classes], axis=-1)

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


def upsample(inputs, out_shape, data_format):
    """
        Upsamples to `out_shape` using nearest neighbor interpolation.
        modified from https://www.kaggle.com/aruchomu/yolo-v3-object-detection-in-tensorflow
    """

    if data_format == 'channels_first':
        inputs = tf.transpose(inputs, [0, 2, 3, 1])
        new_height = out_shape[3]
        new_width = out_shape[2]
    else:
        new_height = out_shape[2]
        new_width = out_shape[1]

    inputs = tf.compat.v1.image.resize_nearest_neighbor(inputs, (new_height, new_width))

    if data_format == 'channels_first':
        inputs = tf.transpose(inputs, [0, 3, 1, 2])

    return inputs


def batch_norm(input_layer, training, data_format):
    """Performs a batch normalization using a standard set of parameters.
    modified from https://www.kaggle.com/aruchomu/yolo-v3-object-detection-in-tensorflow
    """
    return tf.keras.layers.BatchNormalization(axis=1 if data_format == 'channels_first' else 3,
                                              momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON,
                                              scale=True, training=training)(input_layer)


def fixed_padding(input_layer, kernel_size, data_format):
    """ResNet implementation of fixed padding.

    Pads the input along the spatial dimensions independently of input size.

    Args:
        input_layer: Tensor input to be padded.
        kernel_size: The kernel to be used in the conv2d or max_pool2d.
        data_format: The input format.
    Returns:
        A tensor with the same format as the input.

    modified from https://www.kaggle.com/aruchomu/yolo-v3-object-detection-in-tensorflow
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if data_format == 'channels_first':
        padded_inputs = tf.keras.layers.ZeroPadding2D(padding=[[0, 0], [0, 0],
                                                               [pad_beg, pad_end],
                                                               [pad_beg, pad_end]])(input_layer)
    else:
        padded_inputs = tf.keras.layers.ZeroPadding2D(padding=[[0, 0], [pad_beg, pad_end],
                                                               [pad_beg, pad_end], [0, 0]])(input_layer)
    return padded_inputs


def conv2d_fixed_padding(input_layer, filters, kernel_size, data_format, strides=1):
    """
        Strided 2-D convolution with explicit padding.
        modified from https://www.kaggle.com/aruchomu/yolo-v3-object-detection-in-tensorflow
    """
    if strides > 1:
        inputs = fixed_padding(input_layer=input_layer, kernel_size=kernel_size, data_format=data_format)

    return tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                  strides=strides, padding=('SAME' if strides == 1 else 'VALID'),
                                  use_bias=False, data_format=data_format)(input_layer)


def build_boxes(inputs):
    """
        Computes top left and bottom right points of the boxes.
        from https://www.kaggle.com/aruchomu/yolo-v3-object-detection-in-tensorflow
    """
    center_x, center_y, width, height, confidence, classes = \
        tf.split(inputs, [1, 1, 1, 1, 1, -1], axis=-1)

    top_left_x = center_x - width / 2
    top_left_y = center_y - height / 2
    bottom_right_x = center_x + width / 2
    bottom_right_y = center_y + height / 2

    boxes = tf.concat([top_left_x, top_left_y,
                       bottom_right_x, bottom_right_y,
                       confidence, classes], axis=-1)

    return boxes


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


def model_yolo_v3(n_classes, model_size, max_output_size, iou_threshold,
                  confidence_threshold, training, data_format=None):
    """
        Creates the model.

            Args:
                n_classes: Number of class labels.
                model_size: The input size of the model.
                max_output_size: Max number of boxes to be selected for each class.
                iou_threshold: Threshold for the IOU.
                confidence_threshold: Threshold for the confidence score.
                data_format: The input format.
                training: Boolean; whether or not the model is for training

            Returns:
                None.

            modified from https://www.kaggle.com/aruchomu/yolo-v3-object-detection-in-tensorflow
    """

    if not data_format:
        if tf.test.is_built_with_cuda():
            data_format = 'channels_first'
        else:
            data_format = 'channels_last'

    if data_format == 'channels_first':
        inputs = tf.transpose(inputs, [0, 3, 1, 2])

    inputs = inputs / 255

    route1, route2, inputs = darknet_53_block(inputs, training=training,
                                              data_format=self.data_format)

    route, inputs = yolo_convolution_block(
        inputs, filters=512, training=training,
        data_format=self.data_format)
    detect1 = yolo_layer(inputs, n_classes=self.n_classes,
                         anchors=_ANCHORS[6:9],
                         img_size=self.model_size,
                         data_format=self.data_format)

    inputs = conv2d_fixed_padding(route, filters=256, kernel_size=1,
                                  data_format=self.data_format)
    inputs = batch_norm(inputs, training=training,
                        data_format=self.data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)
    upsample_size = route2.get_shape().as_list()
    inputs = upsample(inputs, out_shape=upsample_size,
                      data_format=self.data_format)
    axis = 1 if self.data_format == 'channels_first' else 3
    inputs = tf.concat([inputs, route2], axis=axis)
    route, inputs = yolo_convolution_block(
        inputs, filters=256, training=training,
        data_format=self.data_format)
    detect2 = yolo_layer(inputs, n_classes=self.n_classes,
                         anchors=_ANCHORS[3:6],
                         img_size=self.model_size,
                         data_format=self.data_format)

    inputs = conv2d_fixed_padding(route, filters=128, kernel_size=1,
                                  data_format=self.data_format)
    inputs = batch_norm(inputs, training=training,
                        data_format=self.data_format)
    inputs = tf.nn.leaky_relu(inputs, alpha=_LEAKY_RELU)
    upsample_size = route1.get_shape().as_list()
    inputs = upsample(inputs, out_shape=upsample_size,
                      data_format=self.data_format)
    inputs = tf.concat([inputs, route1], axis=axis)
    route, inputs = yolo_convolution_block(
        inputs, filters=128, training=training,
        data_format=self.data_format)
    detect3 = yolo_layer(inputs, n_classes=self.n_classes,
                         anchors=_ANCHORS[0:3],
                         img_size=self.model_size,
                         data_format=self.data_format)

    inputs = tf.concat([detect1, detect2, detect3], axis=1)

    inputs = build_boxes(inputs)

    boxes_dicts = non_max_suppression(
        inputs, n_classes=self.n_classes,
        max_output_size=self.max_output_size,
        iou_threshold=self.iou_threshold,
        confidence_threshold=self.confidence_threshold)

    return boxes_dicts
