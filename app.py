import streamlit as st
import cv2
import numpy as np

import matplotlib.pyplot as plt
import os

from PIL import Image, ImageOps
from numpy import argmax

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

st.title("MNIST Classification")
st.header("Please input an image to be classified:")
st.text("Created by MML")

@st.cache(allow_output_mutation=True)

def load_image(image):
    # img = load_img(filename, grayscale=True, target_size=(28,28))
    # img = img_to_array(img)
    # img = img.reshape(1, 28, 28, 1)
    # img = img.astype('float32')
    # img = img / 255.0
    
    size = (28, 28)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    #turn the image into a numpy array
    image_array = np.asarray(image)
    image_array = image_array.reshape(1, 28, 28, 1)
    normalized_image_array = (image_array.astype(np.float32) / 255)
    return normalized_image_array

def teachable_machine_classification(img, weights_file):
    img = load_image(img)
    model = keras.models.load_model(weights_file)
    # model.fit(trainX, trainY, epochs=10, batch_size=32, verbose=0)
    predict_value = model.predict(img)
    prediction=predict_value.round()
    digit = argmax(predict_value)
    return digit, prediction


uploaded_file = st.file_uploader("Choose an Number Image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded file', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    digit, perc = teachable_machine_classification(image, 'number_recognize.h5')
    # st.write(digit)
    st.write("It's <", digit , ">. confidence level:", perc)
