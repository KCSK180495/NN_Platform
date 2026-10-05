"""Models for NN Training Studio."""

import keras
from keras import layers
from keras import models
import numpy as np
import tensorflow as tf
from nn_training_studio.constants import (
    IMAGE_MODEL_CNN,
    IMAGE_MODEL_MOBILENET,
    MODEL_TYPE_SAFE_AI_LEGACY,
    MODEL_TYPE_SAFE_CUSTOM,
    SUPPORTED_IMAGE_MODEL_TYPES,
    TASK_AUTOENCODER,
    TASK_CLASSIFICATION,
)


def build_optimizer(optimizer_name, learning_rate):
    if optimizer_name == "Adam":
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)

    if optimizer_name == "SGD":
        return tf.keras.optimizers.SGD(learning_rate=learning_rate)

    if optimizer_name == "RMSprop":
        return tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

    if optimizer_name == "Nadam":
        return tf.keras.optimizers.Nadam(learning_rate=learning_rate)

    raise ValueError("Unsupported optimizer.")


def prepare_training_targets(y, loss_name, output_units):
    if loss_name == "sparse_categorical_crossentropy":
        return y

    if loss_name == "categorical_crossentropy":
        return tf.keras.utils.to_categorical(y, num_classes=output_units)

    if loss_name == "binary_crossentropy":
        return y.astype("float32")

    if loss_name in ["mean_squared_error", "mean_absolute_error"]:
        if output_units == 1:
            return y.astype("float32")
        return tf.keras.utils.to_categorical(y, num_classes=output_units)

    return y


def prediction_to_class(y_prob, output_units):
    if output_units == 1:
        y_prob = y_prob.reshape(-1)
        return (y_prob >= 0.5).astype(int)

    return np.argmax(y_prob, axis=1)


def build_dnn(input_shape, output_units, hidden_activation, output_activation, dropout_rate):
    model_layers = [layers.Input(shape=input_shape)]
    if len(tuple(input_shape)) > 1:
        model_layers.append(layers.Flatten())
    model_layers.extend([
        layers.Dense(128, activation=hidden_activation),
        layers.Dropout(dropout_rate),

        layers.Dense(64, activation=hidden_activation),
        layers.Dropout(dropout_rate),

        layers.Dense(output_units, activation=output_activation)
    ])
    model = models.Sequential(model_layers)

    return model


def build_cnn(input_shape, output_units, hidden_activation, output_activation, dropout_rate):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv1D(64, kernel_size=3, padding="same", activation=hidden_activation),
        layers.MaxPooling1D(pool_size=2),

        layers.Conv1D(128, kernel_size=3, padding="same", activation=hidden_activation),
        layers.GlobalAveragePooling1D(),

        layers.Dense(64, activation=hidden_activation),
        layers.Dropout(dropout_rate),

        layers.Dense(output_units, activation=output_activation)
    ])

    return model


def build_lstm(input_shape, output_units, hidden_activation, output_activation, dropout_rate):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.LSTM(64),

        layers.Dense(64, activation=hidden_activation),
        layers.Dropout(dropout_rate),

        layers.Dense(output_units, activation=output_activation)
    ])

    return model


def build_cnn_lstm(input_shape, output_units, hidden_activation, output_activation, dropout_rate):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv1D(64, kernel_size=3, padding="same", activation=hidden_activation),
        layers.MaxPooling1D(pool_size=2),

        layers.LSTM(64),

        layers.Dense(64, activation=hidden_activation),
        layers.Dropout(dropout_rate),

        layers.Dense(output_units, activation=output_activation)
    ])

    return model


def build_dense_autoencoder(input_shape, hidden_activation, dropout_rate):
    if len(tuple(input_shape)) != 1:
        raise ValueError(
            "Dense Autoencoder requires row-based input. "
            "Choose LSTM Autoencoder for window-based reconstruction."
        )

    feature_count = int(input_shape[0])
    return models.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(128, activation=hidden_activation),
        layers.Dropout(dropout_rate),
        layers.Dense(32, activation=hidden_activation, name="bottleneck"),
        layers.Dense(128, activation=hidden_activation),
        layers.Dense(feature_count, activation="linear")
    ])


def build_lstm_autoencoder(input_shape, hidden_activation, dropout_rate):
    if len(tuple(input_shape)) != 2:
        raise ValueError("LSTM Autoencoder requires window-based 3D input.")

    time_steps, feature_count = input_shape
    inputs = keras.Input(shape=input_shape)
    encoded = layers.LSTM(64, activation="tanh")(inputs)
    encoded = layers.Dropout(dropout_rate)(encoded)
    encoded = layers.Dense(32, activation=hidden_activation, name="bottleneck")(encoded)
    decoded = layers.RepeatVector(time_steps)(encoded)
    decoded = layers.LSTM(64, activation="tanh", return_sequences=True)(decoded)
    outputs = layers.TimeDistributed(
        layers.Dense(feature_count, activation="linear")
    )(decoded)
    return keras.Model(inputs, outputs, name="lstm_autoencoder")


def build_custom_model_from_code(
    custom_code,
    input_shape,
    output_units,
    hidden_activation,
    output_activation,
    dropout_rate
):
    """Execute and validate a user-defined build_custom_model function."""
    if not custom_code or not custom_code.strip():
        raise ValueError("Custom model code is empty.")

    namespace = {
        "tf": tf,
        "keras": keras,
        "layers": layers,
        "models": models,
        "np": np,
    }

    try:
        compiled_code = compile(custom_code, "<custom_model>", "exec")
        exec(compiled_code, namespace)
    except Exception as exc:
        raise ValueError(f"Custom model code could not be loaded: {exc}") from exc

    builder = namespace.get("build_custom_model")

    if not callable(builder):
        raise ValueError(
            "Custom model code must define build_custom_model("
            "input_shape, output_units, hidden_activation, "
            "output_activation, dropout_rate)."
        )

    try:
        model = builder(
            input_shape=input_shape,
            output_units=output_units,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
            dropout_rate=dropout_rate
        )
    except TypeError:
        # Also support users who wrote a positional-argument function.
        model = builder(
            input_shape,
            output_units,
            hidden_activation,
            output_activation,
            dropout_rate
        )
    except Exception as exc:
        raise ValueError(f"Custom model construction failed: {exc}") from exc

    if not isinstance(model, keras.Model):
        raise ValueError("build_custom_model must return a keras.Model.")

    return model


def build_safe_model_from_spec(
    model_spec,
    input_shape,
    output_units,
    output_activation,
    task_type,
):
    """Construct a Keras model from a validated allowlisted layer plan."""
    if not isinstance(model_spec, dict):
        raise ValueError("The safe AI model specification is missing.")

    model = keras.Sequential(name="ai_safe_generated_model")
    model.add(layers.Input(shape=input_shape))

    for index, layer_spec in enumerate(model_spec.get("layers", [])):
        layer_type = layer_spec["layer_type"]
        layer_name = f"safe_{index + 1}_{layer_type.lower()}"
        if layer_type == "Dense":
            model.add(layers.Dense(
                int(layer_spec["units"]),
                activation=layer_spec["activation"],
                name=layer_name,
            ))
        elif layer_type == "Dropout":
            model.add(layers.Dropout(
                float(layer_spec["dropout_rate"]),
                name=layer_name,
            ))
        elif layer_type == "BatchNormalization":
            model.add(layers.BatchNormalization(name=layer_name))
        elif layer_type == "Conv1D":
            model.add(layers.Conv1D(
                filters=int(layer_spec["filters"]),
                kernel_size=int(layer_spec["kernel_size"]),
                strides=int(layer_spec["strides"]),
                padding=layer_spec["padding"],
                activation=layer_spec["activation"],
                name=layer_name,
            ))
        elif layer_type in ("MaxPooling1D", "AveragePooling1D"):
            pool_class = (
                layers.MaxPooling1D
                if layer_type == "MaxPooling1D"
                else layers.AveragePooling1D
            )
            pool_padding = (
                "same"
                if layer_spec["padding"] == "causal"
                else layer_spec["padding"]
            )
            model.add(pool_class(
                pool_size=int(layer_spec["pool_size"]),
                padding=pool_padding,
                name=layer_name,
            ))
        elif layer_type == "GlobalAveragePooling1D":
            model.add(layers.GlobalAveragePooling1D(name=layer_name))
        elif layer_type == "GlobalMaxPooling1D":
            model.add(layers.GlobalMaxPooling1D(name=layer_name))
        elif layer_type == "Flatten":
            model.add(layers.Flatten(name=layer_name))
        elif layer_type in ("LSTM", "GRU"):
            recurrent_class = (
                layers.LSTM if layer_type == "LSTM" else layers.GRU
            )
            recurrent_layer = recurrent_class(
                int(layer_spec["units"]),
                activation=layer_spec["activation"],
                return_sequences=bool(layer_spec["return_sequences"]),
                name=layer_name + "_core",
            )
            if layer_spec["bidirectional"]:
                recurrent_layer = layers.Bidirectional(
                    recurrent_layer,
                    name=layer_name,
                )
            model.add(recurrent_layer)
        else:
            raise ValueError(
                f"Unsupported layer in safe AI model: {layer_type}"
            )

    if task_type == TASK_AUTOENCODER:
        feature_count = int(input_shape[-1])
        if len(input_shape) == 1:
            model.add(layers.Dense(
                feature_count,
                activation="linear",
                name="reconstruction_output",
            ))
        else:
            model.add(layers.TimeDistributed(
                layers.Dense(feature_count, activation="linear"),
                name="reconstruction_output",
            ))
        expected_output_shape = tuple(input_shape)
    else:
        model.add(layers.Dense(
            int(output_units),
            activation=output_activation,
            name="task_output",
        ))
        expected_output_shape = (int(output_units),)

    actual_output_shape = tuple(model.output_shape[1:])
    if actual_output_shape != expected_output_shape:
        raise ValueError(
            "The generated model has an incompatible output shape. "
            f"Expected {expected_output_shape}, received {actual_output_shape}."
        )
    return model


def build_image_classification_model(
    model_type,
    input_shape,
    output_units,
    dropout_rate,
    augmentation_enabled,
    pretrained_weights,
):
    """Build a 2D image classifier with preprocessing stored in the model."""
    if model_type not in SUPPORTED_IMAGE_MODEL_TYPES:
        raise ValueError(f"Unsupported image model type: {model_type}")
    if len(tuple(input_shape)) != 3:
        raise ValueError(
            "Image models require input shape (height, width, channels)."
        )

    inputs = keras.Input(shape=input_shape, name="image")
    x = inputs
    if augmentation_enabled:
        augmentation = keras.Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.05),
                layers.RandomZoom(0.10),
                layers.RandomContrast(0.10),
            ],
            name="image_augmentation",
        )
        x = augmentation(x)

    if model_type == IMAGE_MODEL_CNN:
        x = layers.Rescaling(1.0 / 255.0, name="rescale_0_1")(x)
        for filters in (32, 64, 128):
            x = layers.Conv2D(
                filters,
                kernel_size=3,
                padding="same",
                activation="relu",
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(
            output_units,
            activation="softmax",
            name="class_output",
        )(x)
        return keras.Model(inputs, outputs, name="image_cnn")

    if int(input_shape[-1]) != 3:
        raise ValueError(
            "MobileNetV2 and EfficientNetB0 require RGB images. "
            "Select RGB colour mode or use Image CNN."
        )

    weights = "imagenet" if pretrained_weights else None
    if model_type == IMAGE_MODEL_MOBILENET:
        x = keras.applications.mobilenet_v2.preprocess_input(x)
        base_model = keras.applications.MobileNetV2(
            include_top=False,
            weights=weights,
            input_shape=input_shape,
        )
    else:
        # Keras EfficientNetB0 includes its own input rescaling.
        base_model = keras.applications.EfficientNetB0(
            include_top=False,
            weights=weights,
            input_shape=input_shape,
        )

    # Freeze pretrained features for a stable first training stage. A model
    # initialized without ImageNet weights is trained end-to-end.
    base_model.trainable = not pretrained_weights
    x = (
        base_model(x, training=False)
        if pretrained_weights
        else base_model(x)
    )
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(
        output_units,
        activation="softmax",
        name="class_output",
    )(x)
    return keras.Model(
        inputs,
        outputs,
        name=(
            "mobilenet_v2_classifier"
            if model_type == IMAGE_MODEL_MOBILENET
            else "efficientnet_b0_classifier"
        ),
    )


def build_selected_model(
    model_type,
    input_shape,
    output_units,
    hidden_activation,
    output_activation,
    dropout_rate,
    custom_model_code=None,
    safe_model_spec=None,
    task_type=TASK_CLASSIFICATION
):
    if model_type == "Dense Autoencoder":
        return build_dense_autoencoder(
            input_shape,
            hidden_activation,
            dropout_rate
        )

    if model_type == "LSTM Autoencoder":
        return build_lstm_autoencoder(
            input_shape,
            hidden_activation,
            dropout_rate
        )

    if model_type == "DNN":
        return build_dnn(
            input_shape,
            output_units,
            hidden_activation,
            output_activation,
            dropout_rate
        )

    if model_type == "CNN":
        return build_cnn(
            input_shape,
            output_units,
            hidden_activation,
            output_activation,
            dropout_rate
        )

    if model_type == "LSTM":
        return build_lstm(
            input_shape,
            output_units,
            hidden_activation,
            output_activation,
            dropout_rate
        )

    if model_type == "CNN-LSTM":
        return build_cnn_lstm(
            input_shape,
            output_units,
            hidden_activation,
            output_activation,
            dropout_rate
        )

    if model_type == "Custom Python Model":
        return build_custom_model_from_code(
            custom_code=custom_model_code,
            input_shape=input_shape,
            output_units=output_units,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
            dropout_rate=dropout_rate
        )

    if model_type in (MODEL_TYPE_SAFE_CUSTOM, MODEL_TYPE_SAFE_AI_LEGACY):
        return build_safe_model_from_spec(
            model_spec=safe_model_spec,
            input_shape=input_shape,
            output_units=output_units,
            output_activation=output_activation,
            task_type=task_type,
        )

    raise ValueError("Unsupported model type.")
