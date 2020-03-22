import sys
import matplotlib.pyplot as plt
import altair as alt
import os

sys.path.append("learning_to_abstain")

import streamlit as st
import predictor

st.title("Classify Flowers")

file = st.file_uploader("Upload file", type=["jpg"])

ood_predictor_config_path = (
    "ood_config.yaml"
    if os.path.isfile("ood_config.yaml")
    else "learning_to_abstain/ood_config.yaml"
)


ood_image_predictor = predictor.ImagePredictor.init_from_config_url(
    ood_predictor_config_path
)

if file:
    ood_pred, arr = ood_image_predictor.predict_from_file(file)

    plt.imshow(arr)
    plt.axis("off")
    st.pyplot()
    st.write("Outlier Exposed")
    st.write(ood_pred)
