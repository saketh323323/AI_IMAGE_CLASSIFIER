from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import base64
import cv2

app = Flask(__name__)

# Load the model from the .pkl file
with open('classifier.pkl', 'rb') as f:
    model = pickle.load(f)

# Define your prediction endpoint
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input image from the request
    print("qjkdbwkefwk")
    file = request.files['file']
    img_bytes = file.read()

    # Convert image bytes to numpy array
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Resize the image
    resize = cv2.resize(img, (256, 256))

    # Preprocess your image here if needed
    # For example, normalize
    # input_data = resize / 255.0

    # Make predictions
    yhat = model.predict(np.expand_dims(resize/255, 0))

    # Assuming yhat is the prediction result, you can return it as JSON
    prediction = {'prediction': yhat.tolist()}

    # If prediction > 0.5, consider it as 'fake'
    if yhat <= 0.5:
        prediction1 = 'fake'
    else:
        prediction1='real'

    return prediction1
