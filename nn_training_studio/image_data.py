"""Image data for NN Training Studio."""

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from nn_training_studio.constants import (
    DATA_MODE_IMAGE,
    SUPPORTED_IMAGE_EXTENSIONS,
)


def scan_image_dataset(
    directory,
    minimum_class_count=2,
    minimum_images_per_class=3,
):
    """
    Scan an image-classification folder.

    Expected layout:
        dataset/
            class_a/image_001.jpg
            class_a/image_002.png
            class_b/image_003.jpg

    Images may be placed in nested folders below each class folder. The direct
    child folder name is always used as the class label.
    """
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("The selected image dataset folder does not exist.")

    class_directories = sorted(
        path for path in root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    if len(class_directories) < minimum_class_count:
        raise ValueError(
            f"The selected folder requires at least {minimum_class_count} "
            "class folder(s)."
        )

    records = []
    class_names = []
    for class_directory in class_directories:
        image_paths = sorted(
            path for path in class_directory.rglob("*")
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        )
        if not image_paths:
            continue
        class_index = len(class_names)
        class_names.append(class_directory.name)
        for image_path in image_paths:
            records.append({
                "file_path": str(image_path),
                "class_name": class_directory.name,
                "class_index": class_index,
                "extension": image_path.suffix.lower(),
            })

    if len(class_names) < minimum_class_count:
        raise ValueError(
            f"At least {minimum_class_count} class folder(s) must contain "
            "supported images "
            "(.jpg, .jpeg, .png, .bmp, or .gif)."
        )

    records_df = pd.DataFrame.from_records(records)
    counts = records_df["class_name"].value_counts()
    too_small = counts[counts < minimum_images_per_class]
    if not too_small.empty:
        details = ", ".join(
            f"{name}={int(count)}" for name, count in too_small.items()
        )
        raise ValueError(
            f"Each class needs at least {minimum_images_per_class} image(s). "
            f"Too small: {details}"
        )

    return records_df, class_names


def build_image_dataset_profile(records_df, class_names, root_directory):
    class_counts = (
        records_df.groupby("class_name", sort=False)
        .size()
        .reindex(class_names)
    )
    extension_counts = records_df["extension"].value_counts().sort_index()
    return {
        "analysis_version": 2,
        "data_mode": DATA_MODE_IMAGE,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "root_directory": str(root_directory),
        "image_count": int(len(records_df)),
        "class_count": int(len(class_names)),
        "class_names": list(class_names),
        "class_counts": {
            str(name): int(count)
            for name, count in class_counts.items()
        },
        "extension_counts": {
            str(extension): int(count)
            for extension, count in extension_counts.items()
        },
    }


def split_image_records(
    records_df,
    test_fraction,
    validation_fraction,
    random_seed=42,
):
    """
    Split every class independently so all three sets contain every class.

    validation_fraction is applied to the records remaining after the test set
    is selected, matching Keras validation_split semantics.
    """
    if not 0.0 < test_fraction < 0.9:
        raise ValueError("Image test fraction must be between 0 and 0.9.")
    if not 0.0 < validation_fraction < 0.8:
        raise ValueError(
            "Image validation fraction must be between 0 and 0.8."
        )

    rng = np.random.default_rng(random_seed)
    train_parts = []
    validation_parts = []
    test_parts = []

    for _, class_records in records_df.groupby("class_index", sort=True):
        shuffled_positions = rng.permutation(len(class_records))
        shuffled = class_records.iloc[shuffled_positions].reset_index(drop=True)
        sample_count = len(shuffled)
        if sample_count < 3:
            raise ValueError(
                "Every image class requires at least three photos."
            )

        test_count = max(1, int(round(sample_count * test_fraction)))
        test_count = min(test_count, sample_count - 2)
        remaining_count = sample_count - test_count
        validation_count = max(
            1,
            int(round(remaining_count * validation_fraction)),
        )
        validation_count = min(validation_count, remaining_count - 1)

        test_parts.append(shuffled.iloc[:test_count])
        validation_parts.append(
            shuffled.iloc[test_count:test_count + validation_count]
        )
        train_parts.append(
            shuffled.iloc[test_count + validation_count:]
        )

    train_df = pd.concat(train_parts, ignore_index=True)
    validation_df = pd.concat(validation_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)
    train_df = train_df.iloc[rng.permutation(len(train_df))].reset_index(
        drop=True
    )
    validation_df = validation_df.iloc[
        rng.permutation(len(validation_df))
    ].reset_index(drop=True)
    test_df = test_df.iloc[rng.permutation(len(test_df))].reset_index(
        drop=True
    )
    return train_df, validation_df, test_df


def make_image_tf_dataset(
    records_df,
    image_height,
    image_width,
    color_mode,
    batch_size,
    shuffle=False,
):
    """Create a memory-efficient image dataset that preserves record order."""
    import tensorflow as tf

    channels = 1 if color_mode == "Grayscale" else 3
    paths = records_df["file_path"].astype(str).to_numpy()
    labels_array = records_df["class_index"].astype(np.int32).to_numpy()
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels_array))

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=max(1, len(records_df)),
            seed=42,
            reshuffle_each_iteration=True,
        )

    def load_image(path_value, label_value):
        encoded = tf.io.read_file(path_value)
        image = tf.io.decode_image(
            encoded,
            channels=channels,
            expand_animations=False,
        )
        image.set_shape([None, None, channels])
        image = tf.image.resize(
            image,
            [image_height, image_width],
            method="bilinear",
            antialias=True,
        )
        image = tf.cast(image, tf.float32)
        return image, label_value

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=not shuffle,
    )
    dataset = dataset.batch(batch_size)
    return dataset.prefetch(tf.data.AUTOTUNE)
