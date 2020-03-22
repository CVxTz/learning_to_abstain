import sys
import matplotlib.pyplot as plt
import altair as alt
import os

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

ood_predictor_config_path = (
    "ood_config.yaml"
    if os.path.isfile("ood_config.yaml")
    else "learning_to_abstain/ood_config.yaml"
)


image_predictor = predictor.ImagePredictor.init_from_config_url(predictor_config_path)

ood_image_predictor = predictor.ImagePredictor.init_from_config_url(ood_predictor_config_path)

if file:
    pred, arr = image_predictor.predict_from_file(file)
    ood_pred, _ = image_predictor.predict_from_file(file)

    plt.imshow(arr)
    plt.axis("off")
    st.pyplot()
    st.write("Outlier Exposed")
    st.write(ood_pred)
    st.write("Regular ERM")
    st.write(pred)

