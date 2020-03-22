import argparse
import json

import numpy as np
import pandas as pd
import yaml
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

from preprocessing_utilities import (
    read_img_from_path,
    resize_img,
    read_from_file,
)
from utils import download_model


class ImagePredictor:
    def __init__(
        self,
        model_path,
        resize_size,
        class_display_names,
        pre_processing_function=preprocess_input,
    ):
        self.model_path = model_path
        self.pre_processing_function = pre_processing_function
        self.model = load_model(self.model_path, compile=False)
        self.resize_size = resize_size
        self.class_display_names = class_display_names

    @classmethod
    def init_from_config_path(cls, config_path):
        with open(config_path, "r") as f:
            config = yaml.load(f, yaml.SafeLoader)
        predictor = cls(
            model_path=config["model_path"],
            resize_size=config["resize_shape"],
            class_display_names=config["class_display_names"],
        )
        return predictor

    @classmethod
    def init_from_config_url(cls, config_path):
        with open(config_path, "r") as f:
            config = yaml.load(f, yaml.SafeLoader)

        download_model(
            config["model_url"], config["model_path"], config["model_sha256"],
        )

        return cls.init_from_config_path(config_path)

    def predict_from_array(self, arr):
        arr = resize_img(arr, h=self.resize_size[0], w=self.resize_size[1])
        arr = self.pre_processing_function(arr)
        pred = self.model.predict(arr[np.newaxis, ...]).ravel()
        label, max_score = np.argmax(pred), np.max(pred)
        label_name = (
            self.class_display_names[label] if max_score > 0.15 else "Don't Know"
        )
        # label = label + 1  # labels are 1-indexed while arrays are 0-indexed ..;
        return {"label": label_name, "score": float(max_score)}

    def predict_from_path(self, path):
        arr = read_img_from_path(path)
        return self.predict_from_array(arr)

    def predict_from_file(self, file_object):
        arr = read_from_file(file_object)
        return self.predict_from_array(arr), arr


if __name__ == "__main__":
    """
    python predictor.py --predictor_config "config.yaml"

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictor_config_path", help="predictor_config_path", default="config.yaml",
    )

    args = parser.parse_args()

    predictor_config_path = args.predictor_config_path

    predictor = ImagePredictor.init_from_config_url(predictor_config_path)

    pred = predictor.predict_from_file(open("../example/data/image_00005.jpg", "rb"))

    print(pred)
