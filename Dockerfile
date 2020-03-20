FROM python:3.6-slim
COPY learning_to_abstain/main.py learning_to_abstain/preprocessing_utilities.py /deploy/
COPY learning_to_abstain/predictor.py learning_to_abstain/utils.py /deploy/
COPY learning_to_abstain/config.yaml /deploy/
# Download from https://github.com/CVxTz/learning_to_abstain/releases
COPY image_tag_suggestion/labels.json /deploy/
# Download from https://github.com/CVxTz/learning_to_abstain/releases
COPY requirements.txt /deploy/
WORKDIR /deploy/
RUN pip install -r requirements.txt
EXPOSE 8501

ENTRYPOINT streamlit run main.py