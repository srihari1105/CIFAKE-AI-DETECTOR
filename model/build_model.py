import tensorflow as tf
from tensorflow.keras import layers, models


IMAGE_SIZE = (32, 32, 3)


def build_model():

    inputs = layers.Input(shape=IMAGE_SIZE)

    # Block 1
    x = layers.Conv2D(
        32, (3, 3),
        padding="same",
        activation="relu"
    )(inputs)

    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(
        32, (3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.15)(x)

    # Block 2
    x = layers.Conv2D(
        64, (3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(
        64, (3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.20)(x)

    # Block 3
    x = layers.Conv2D(
        128, (3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(
        128, (3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Feature representation
    x = layers.GlobalAveragePooling2D()(x)

    # Classification head
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(
        1,
        activation="sigmoid"
    )(x)

    model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name="CIFAKE_Custom_CNN"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3
        ),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc")
        ]
    )

    return model