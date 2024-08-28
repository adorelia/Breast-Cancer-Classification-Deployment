from flask import Flask, render_template, request
import cv2
import numpy as np
import joblib
import os

app = Flask(__name__)

# Ensure the images directory exists
if not os.path.exists('./images'):
    os.makedirs('./images')

# Load the SVM model
svm_model = joblib.load('Mammography_SVM.model')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = os.path.join('./images', imagefile.filename)
    imagefile.save(image_path)

    # Preprocess the image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (64, 64))
    image = image.flatten().reshape(1, -1)  # Flatten the image to match SVM input
    image = image / 255.0  # Scale the image

    # Predict using the SVM model
    prediction = svm_model.predict(image)
    prediction_proba = svm_model.predict_proba(image)
    class_label = prediction[0]
    confidence = max(prediction_proba[0])

    classification = f'{class_label} ({confidence * 100:.2f}%)'

    return render_template('index.html', prediction=classification)

if __name__ == '__main__':
    app.run(port=3000, debug=True)
