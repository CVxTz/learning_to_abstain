import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from models import get_model_classification
from training_utilities import (
    df_to_list_samples,
    batch_generator,
)


def train_from_csv(csv_train, training_config_path):
    with open(training_config_path, "r") as f:
        training_config = yaml.load(f, yaml.SafeLoader)

    train = pd.read_csv(Path(training_config["data_path"]) / csv_train)

    train_samples, val_samples = df_to_list_samples(train, fold="flowers")

    model = get_model_classification()

    print("train_samples", len(train_samples))
    print("val_samples", len(val_samples))

    train_gen = batch_generator(
        train_samples,
        resize_size=training_config["resize_shape"],
        augment=training_config["use_augmentation"],
        base_path=training_config["data_path"],
    )
    val_gen = batch_generator(
        val_samples,
        resize_size=training_config["resize_shape"],
        base_path=training_config["data_path"],
    )

    checkpoint = ModelCheckpoint(
        training_config["model_path"],
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        mode="min",
    )
    reduce = ReduceLROnPlateau(monitor="val_loss", mode="min", patience=15, min_lr=1e-7)
    early = EarlyStopping(monitor="val_loss", mode="min", patience=30)

    # try:
    #     model.load_weights(training_config["model_path"])
    # except:
    #     print("No model to load")

    model.fit_generator(
        train_gen,
        steps_per_epoch=100,
        validation_data=val_gen,
        validation_steps=50,
        epochs=1000,
        callbacks=[checkpoint, reduce, early],
        use_multiprocessing=True,
        workers=8,
        verbose=2,
    )

    # for _ in range(training_config["epochs"]):
    #     model.fit_generator(
    #         train_gen,
    #         steps_per_epoch=100,
    #         validation_data=val_gen,
    #         validation_steps=50,
    #         epochs=1,
    #         callbacks=[checkpoint, reduce, early],
    #         use_multiprocessing=True,
    #         workers=8,
    #         verbose=2,
    #     )
    #     X, Y = next(train_gen)
    #
    #     pred = model.predict(X)
    #
    #     print(pred[3, -1])
    #
    #     for a, b in zip(pred[3, ...].tolist(), Y[3, ...].tolist()):
    #         print(round(a, 2), b)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_train", help="csv_train", default="imagelabels.csv",  #  oidv6-train
    )
    parser.add_argument(
        "--training_config_path",
        help="training_config_path",
        default="../example/training_config.yaml",
    )
    args = parser.parse_args()

    csv_train = args.csv_train

    training_config_path = args.training_config_path

    train_from_csv(
        csv_train=csv_train, training_config_path=training_config_path,
    )
