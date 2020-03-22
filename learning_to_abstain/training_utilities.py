from collections import namedtuple
from pathlib import Path
from random import shuffle

import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from preprocessing_utilities import (
    read_img_from_path,
    resize_img,
)

SampleFromPath = namedtuple("Sample", ["path", "label"])
import imgaug.augmenters as iaa


def chunks(seq, size):
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


def get_seq():
    sometimes = lambda aug: iaa.Sometimes(0.1, aug)
    seq = iaa.Sequential(
        [
            sometimes(iaa.Affine(scale={"x": (0.8, 1.2)})),
            sometimes(iaa.Fliplr(p=0.5)),
            sometimes(iaa.Affine(scale={"y": (0.8, 1.2)})),
            sometimes(iaa.Affine(translate_percent={"x": (-0.2, 0.2)})),
            sometimes(iaa.Affine(translate_percent={"y": (-0.2, 0.2)})),
            sometimes(iaa.Affine(rotate=(-20, 20))),
            sometimes(iaa.Affine(shear=(-20, 20))),
            sometimes(iaa.AdditiveGaussianNoise(scale=0.07 * 255)),
            sometimes(iaa.GaussianBlur(sigma=(0, 3.0))),
        ],
        random_order=True,
    )
    return seq


def batch_generator(
    list_samples,
    batch_size=32,
    pre_processing_function=None,
    resize_size=(128, 128),
    augment=False,
    n_label=102,
    base_path="",
):
    seq = get_seq()
    pre_processing_function = (
        pre_processing_function
        if pre_processing_function is not None
        else preprocess_input
    )
    while True:
        shuffle(list_samples)
        for batch_samples in chunks(list_samples, size=batch_size):
            images = [
                read_img_from_path(base_path + sample.path) for sample in batch_samples
            ]

            if augment:
                images = seq.augment_images(images=images)

            images = [resize_img(x, h=resize_size[0], w=resize_size[1]) for x in images]

            images = [pre_processing_function(a) for a in images]
            targets = [sample.label for sample in batch_samples]

            X = np.array(images)
            Y = np.array(targets) - 1
            Y = np.eye(n_label)[Y, :]
            yield X, Y


def batch_ood_generator(
    list_samples,
    ood_paths,
    batch_size=32,
    pre_processing_function=None,
    resize_size=(128, 128),
    augment=False,
    n_label=102,
    base_path="",
):
    seq = get_seq()
    pre_processing_function = (
        pre_processing_function
        if pre_processing_function is not None
        else preprocess_input
    )
    while True:
        shuffle(list_samples)
        shuffle(ood_paths)
        for batch_samples, ood_samples in zip(
            chunks(list_samples, size=batch_size // 2),
            chunks(ood_paths, size=batch_size // 2),
        ):
            images = [
                read_img_from_path(base_path + sample.path) for sample in batch_samples
            ]

            ood_images = [read_img_from_path(sample.path) for sample in ood_samples]

            if augment:
                images = seq.augment_images(images=images)
                ood_images = seq.augment_images(images=ood_images)

            images = [resize_img(x, h=resize_size[0], w=resize_size[1]) for x in images]
            ood_images = [
                resize_img(x, h=resize_size[0], w=resize_size[1]) for x in ood_images
            ]

            images = [pre_processing_function(a) for a in images]
            ood_images = [pre_processing_function(a) for a in ood_images]

            targets = [sample.label for sample in batch_samples]

            X = np.array(images + ood_images)
            Y1 = np.array(targets) - 1
            Y1 = np.eye(n_label)[Y1, :].tolist()
            Y2 = [[1 / n_label for _ in range(n_label)]] * len(ood_samples)
            Y = np.array(Y1 + Y2)
            yield X, Y


def df_to_list_samples(df, fold):
    image_name_col = "ImageID"
    label_col = "label"

    paths = df[image_name_col].apply(lambda x: str(Path(fold) / (x + ".jpg"))).tolist()
    list_labels = df[label_col].values.tolist()

    samples = [
        SampleFromPath(path=path, label=label)
        for path, label in zip(paths, list_labels)
    ]

    train_samples = [
        a for a in samples if int(a.path.split("_")[-1].replace(".jpg", "")) % 5 != 0
    ]
    val_samples = [
        a for a in samples if int(a.path.split("_")[-1].replace(".jpg", "")) % 5 == 0
    ]

    return train_samples, val_samples


def get_ood_samples(path):
    paths = Path(path).glob("*.jpg")
    samples = [SampleFromPath(path=str(p), label=None) for p in paths]
    return samples


if __name__ == "__main__":
    import yaml
    import pandas as pd

    config_path = Path("../example/training_config.yaml")

    with open(config_path, "r") as f:
        config = yaml.load(f, yaml.SafeLoader)

    df = pd.read_csv(Path(config["data_path"]) / "imagelabels.csv")

    train_samples, val_samples = df_to_list_samples(df, fold="flowers")

    ood_samples = get_ood_samples(config["ood_data_path"])

    print(train_samples)
    print(ood_samples)
    print(max([x.label for x in train_samples]))

    X, Y = next(
        batch_ood_generator(train_samples, ood_samples, base_path=config["data_path"])
    )

    print(X.shape, X)
    print(Y.shape, Y)
