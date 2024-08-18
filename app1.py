import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import joblib
import cv2
import matplotlib.pyplot as plt
from io import BytesIO

# Load the models
crop_model = load_model('cropmodel.keras')
soil_model = load_model('soilmodel.h5')


def encode_soil_type(soil_type):
    soil_alluvial = 0
    soil_black = 0
    soil_clay = 0
    soil_red = 0

    if soil_type == "Alluvial soil" or soil_type == "Alluvial Soil":
        soil_alluvial = 1
    elif soil_type == "Black soil" or soil_type == "Black Soil":
        soil_black = 1
    elif soil_type == "Clay soil" or soil_type == "Clay Soil":
        soil_clay = 1
    elif soil_type == "Red soil" or soil_type == "Red Soil":
        soil_red = 1
    return soil_alluvial, soil_black, soil_clay, soil_red


def make_prediction(image_fp):
    img = cv2.imdecode(np.frombuffer(image_fp, np.uint8), cv2.IMREAD_COLOR)
    im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(im_rgb)
    plt.axis('off')
    plt.show()

    img = cv2.resize(im_rgb, (256, 256))
    img_array = img / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    class_names = ["Alluvial soil", "Clay soil", "Black soil", "Red soil"]

    predictions = soil_model.predict(img_batch)
    predicted_class_index = np.argmax(predictions)
    predicted_value = class_names[predicted_class_index]

    st.write(f"The predicted soil type is {predicted_value}")
    return predicted_value

# Streamlit App
st.title('Soil and Crop Predictor')

# Image upload for soil prediction
uploaded_file = st.file_uploader("Upload an image of the soil", type="jpg")

if uploaded_file is not None:
    image_fp = BytesIO(uploaded_file.read())
    soil_type_input = make_prediction(image_fp.getvalue())
else:
    soil_type_input = "unknown"

# Show other input fields regardless of image upload
soil_alluvial, soil_black, soil_clay, soil_red = encode_soil_type(soil_type_input)

nitrogen = st.number_input("Nitrogen (numeric value):", min_value=0.0, max_value=100.0, value=50.0)
phosphorous = st.number_input("Phosphorous (numeric value):", min_value=0.0, max_value=100.0, value=50.0)
potassium = st.number_input("Potassium (numeric value):", min_value=0.0, max_value=100.0, value=50.0)
temp = st.number_input("Temperature (numeric value):", min_value=0.0, max_value=50.0, value=25.0)
humidity = st.number_input("Humidity (numeric value):", min_value=0.0, max_value=100.0, value=50.0)
ph = st.number_input("pH (numeric value):", min_value=0.0, max_value=14.0, value=7.0)
rainfall = st.number_input("Rainfall (numeric value):", min_value=0.0, max_value=1000.0, value=500.0)

new_data = np.array([[nitrogen, phosphorous, potassium, temp, humidity, ph, rainfall, soil_alluvial, soil_black, soil_clay, soil_red]])

scaler = joblib.load('scaler.pkl')
new_data_preprocessed = scaler.transform(new_data)

index_to_crop = {
    0: 'apple', 1: 'banana', 2: 'blackgram', 3: 'chickpea', 4: 'coconut',
    5: 'coffee', 6: 'cotton', 7: 'grapes', 8: 'jute', 9: 'kidneybeans',
    10: 'lentil', 11: 'maize', 12: 'mango', 13: 'mothbeans', 14: 'mungbean',
    15: 'muskmelon', 16: 'orange', 17: 'papaya', 18: 'pigeonpeas',
    19: 'pomegranate', 20: 'rice', 21: 'watermelon'
}

if st.button('Predict Crop'):
    predictions = crop_model.predict(new_data_preprocessed)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_crop = index_to_crop.get(predicted_class_index, 'Unknown')
    st.write(f'Predicted crop: {predicted_crop}')
