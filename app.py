import streamlit as st
import cv2
import numpy as np
import joblib

# Load the SVM model
svm_model = joblib.load('Mammography_SVM.model')

def preprocess_image(image):
    # Resize the image to the size expected by the model
    image = cv2.resize(image, (64, 64))
    # Flatten the image
    image = image.reshape(1, -1)
    # Normalize the image
    image = image / 255.0
    return image

def predict(image):
    # Preprocess the image
    image = preprocess_image(image)
    # Predict using the SVM model
    prediction = svm_model.predict(image)
    prediction_proba = svm_model.predict_proba(image)
    class_label = prediction[0]
    confidence = max(prediction_proba[0])
    return class_label, confidence

# Streamlit app
st.title("Mammography Classification")

# Image upload
uploaded_file = st.file_uploader("Choose a mammography image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Display the uploaded image
    st.image(image, channels="BGR", caption="Uploaded Image", use_column_width=True)

    # Make a prediction
    class_label, confidence = predict(image)

    # Display the prediction
    st.write(f"Prediction: **{class_label}**")
    st.write(f"Confidence: **{confidence * 100:.2f}%**")