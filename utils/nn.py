import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    add,
    concatenate,
    Conv1D,
    Conv2D,
    Dropout,
    BatchNormalization,
    Flatten,
    MaxPooling2D,
    AveragePooling1D,
    AveragePooling2D,
    Activation,
    Dropout,
    Reshape,
)
from tensorflow.keras.callbacks import EarlyStopping


def train_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    early_stopping=True,
    min_delta=0.001,
    patience=3,
    batch_size=128,
    epochs=20,
    is_shuffle=True,
    verbose=1,
):
    """
    Train an array of models on the same dataset.
    We use early termination to speed up training.
    """

    # resulting_val_acc = []
    # record_result = []
    # print("Training model ", n)

    if early_stopping:
        model.fit(
            X_train,
            y_train,
            validation_data=[X_test, y_test],
            callbacks=[
                EarlyStopping(
                    monitor="val_accuracy", min_delta=min_delta, patience=patience
                )
            ],
            batch_size=batch_size,
            epochs=epochs,
            shuffle=is_shuffle,
            verbose=verbose,
        )
    else:
        model.fit(
            X_train,
            y_train,
            validation_data=[X_test, y_test],
            batch_size=batch_size,
            epochs=epochs,
            shuffle=is_shuffle,
            verbose=verbose,
        )

    # resulting_val_acc.append(model.history.history["val_accuracy"][-1])
    #     record_result.append(
    #         {
    #             "train_acc": model.history.history["accuracy"],
    #             "val_acc": model.history.history["val_accuracy"],
    #             "train_loss": model.history.history["loss"],
    #             "val_loss": model.history.history["val_loss"],
    #         }
    #     )

    # print("pre-train accuracy: ")
    # print(resulting_val_acc)

    # return record_result


def cnn_2layer_fc_model_2d(
    n_classes, n1=128, n2=256, dropout_rate=0.2, input_shape=(28, 28)
) -> Model:
    model_A, x = None, None

    x = Input(input_shape)
    if len(input_shape) == 2:
        y = Reshape((input_shape[0], input_shape[1], 1))(x)
    else:
        y = Reshape(input_shape)(x)
    y = Conv2D(
        filters=n1, kernel_size=(3, 3), strides=1, padding="same", activation=None
    )(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Dropout(dropout_rate)(y)
    y = AveragePooling2D(pool_size=(2, 2), strides=1, padding="same")(y)

    y = Conv2D(
        filters=n2, kernel_size=(3, 3), strides=2, padding="valid", activation=None
    )(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Dropout(dropout_rate)(y)
    # y = AveragePooling2D(pool_size = (2,2), strides = 2, padding = "valid")(y)

    y = Flatten()(y)
    y = Dense(
        units=n_classes,
        activation=None,
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l2(1e-3),
    )(y)
    y = Activation("softmax")(y)

    model_A = Model(inputs=x, outputs=y)

    model_A.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model_A

def cnn_2layer_fc_model_1d(
    n_classes, n1=128, n2=256, dropout_rate=0.2, input_shape=(28,)
) -> Model:
    model_A, x = None, None

    x = Input(input_shape)
    if len(input_shape) == 2:
        y = Reshape((input_shape[0], input_shape[1], 1))(x)
    else:
        y = Reshape(input_shape)(x)
    y = Conv1D(
        filters=n1, kernel_size=(3,), strides=1, padding="same", activation=None
    )(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Dropout(dropout_rate)(y)
    y = AveragePooling1D(pool_size=(2,), strides=1, padding="same")(y)

    y = Conv1D(
        filters=n2, kernel_size=(3,), strides=2, padding="valid", activation=None
    )(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Dropout(dropout_rate)(y)
    # y = AveragePooling2D(pool_size = (2,2), strides = 2, padding = "valid")(y)

    y = Flatten()(y)
    y = Dense(
        units=n_classes,
        activation=None,
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l2(1e-3),
    )(y)
    y = Activation("softmax")(y)

    model_A = Model(inputs=x, outputs=y)

    model_A.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model_A

def remove_last_layer(model, loss="mean_absolute_error"):
    """
    Input: Keras model, a classification model whose last layer is a softmax activation
    Output: Keras model, the same model with the last softmax activation layer removed,
        while keeping the same parameters
    """

    new_model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    new_model.set_weights(model.get_weights())
    new_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=loss)

    return new_model


from tensorflow.keras.saving import serialize_keras_object, deserialize_keras_object
import json


def dict_to_bytes(dic):
    jsonable_dict = {
        "logits": serialize_keras_object(dic["logits"]),
        "classifier": serialize_keras_object(dic["classifier"]),
        "weights": serialize_keras_object(dic["weights"])
    }
    return json.dumps(jsonable_dict).encode()


def bytes_to_dict(bytes):
    dic = json.loads(bytes.decode())

    return {
        "logits": deserialize_keras_object(dic["logits"]), # remove_last_layer(model, loss="mean_absolute_error"),
        "classifier": deserialize_keras_object(dic["classifier"]), # model,
        "weights": deserialize_keras_object(dic["weights"]), # model.get_weights(),  # model.get_weights(),
    }
