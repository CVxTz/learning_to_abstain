import os
import sys

import matplotlib.pyplot as plt

sys.path.append("learning_to_abstain")

import streamlit as st
import predictor

st.title("Classify Flowers")

file = st.file_uploader("Upload file", type=["jpg"])

predictor_config_path = (
    "config.yaml"
    if os.path.isfile("config.yaml")
    else "learning_to_abstain/config.yaml"
)

image_predictor = predictor.ImagePredictor.init_from_config_url(predictor_config_path)

if file:
    pred, arr = image_predictor.predict_from_file(file)
    plt.imshow(arr)
    plt.axis("off")
    st.pyplot()

    st.write(pred)
