import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.layers import (
    Dense,
    Input,
    Dropout,
    Concatenate,
    GlobalMaxPooling2D,
    GlobalAveragePooling2D,
    Convolution2D,
    MaxPooling2D,
)
from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy
from tensorflow.keras.metrics import binary_accuracy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.framework import ops, constant_op
from tensorflow.python.ops import math_ops, array_ops


def custom_acc(y_true, y_pred):
    return math_ops.cast(
        math_ops.equal(
            math_ops.argmax(y_true, axis=-1), math_ops.argmax(y_pred[:, :-1], axis=-1)
        ),
        K.floatx(),
    )


def repeat1d(x, n):
    x = array_ops.expand_dims(x, 1)

    pattern = array_ops.stack([1, n])

    return array_ops.tile(x, pattern)


def abstain_loss(y_true, y_pred):

    abstain = y_pred[:, -1]

    abstain_repeated = repeat1d(abstain, y_pred.shape[-1] - 1)

    new_y_pred = y_pred[:, :-1] + abstain_repeated

    new_y_pred = ops.convert_to_tensor(new_y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    epsilon_ = constant_op.constant(K.epsilon(), dtype=y_pred.dtype.base_dtype)

    return -0.5 * math_ops.reduce_sum(
        y_true * math_ops.log(new_y_pred + epsilon_), axis=-1
    ) - 0.5 * math_ops.reduce_sum(
        y_true * math_ops.log(y_pred[:, :-1] + epsilon_), axis=-1
    )


def get_model_classification(
    input_shape=(None, None, 3),
    model="mobilenet",
    weights="imagenet",
    n_classes=102,
    multi_class=True,
):
    inputs = Input(input_shape)
    if model == "mobilenet":
        base_model = MobileNetV2(
            include_top=False, input_shape=input_shape, weights=weights
        )
    else:
        base_model = ResNet50(
            include_top=False, input_shape=input_shape, weights=weights
        )

    x = base_model(inputs)
    x = Dropout(0.5)(x)
    out1 = GlobalMaxPooling2D()(x)
    out2 = GlobalAveragePooling2D()(x)
    out = Concatenate(axis=-1)([out1, out2])
    out = Dropout(0.5)(out)
    if multi_class:
        out = Dense(n_classes + 1, activation="softmax")(out)
    else:
        out = Dense(n_classes, activation="sigmoid")(out)

    model = Model(inputs, out)
    if multi_class:
        model.compile(optimizer=Adam(0.0001), loss=abstain_loss, metrics=[custom_acc])
    else:
        model.compile(
            optimizer=Adam(0.0001), loss=binary_crossentropy, metrics=[binary_accuracy]
        )

    model.summary()

    return model


# For debugging purposes


def get_model_mnist(
    input_shape=(None, None, 1), n_classes=102,
):
    inputs = Input(input_shape)
    x = Convolution2D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = MaxPooling2D(pool_size=2)(x)
    x = Convolution2D(filters=6, kernel_size=3, activation="relu")(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Convolution2D(filters=6, kernel_size=3, activation="relu")(x)
    x = Dropout(0.5)(x)
    out1 = GlobalMaxPooling2D()(x)
    out2 = GlobalAveragePooling2D()(x)
    out = Concatenate(axis=-1)([out1, out2])
    out = Dropout(0.5)(out)

    out = Dense(n_classes + 1, activation="softmax")(out)

    model = Model(inputs, out)

    model.compile(optimizer=Adam(0.0001), loss=abstain_loss, metrics=[custom_acc])

    model.summary()

    return model


def get_model_mlp(
    input_shape=(2,), n_classes=3,
):
    inputs = Input(input_shape)
    x = Dense(32, activation="relu")(inputs)
    x = Dropout(0.5)(x)
    x = Dense(32, activation="relu")(x)

    out = Dense(n_classes + 1, activation="softmax")(x)

    model = Model(inputs, out)

    model.compile(optimizer=Adam(0.0001), loss=abstain_loss, metrics=[custom_acc])

    model.summary()

    return model
