from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.decomposition import PCA

app = Flask(__name__)

# Load models once at startup
cnn_model = load_model('mnist_cnn.h5')
svm_model = joblib.load('svm_model.pkl')
pca = joblib.load('pca.pkl')

def preprocess_image(image_file, for_cnn=True):
    # Load and convert the image
    image = Image.open(image_file).convert('L').resize((28, 28))
    image_array = np.array(image)

    if for_cnn:
        image_array = image_array / 255.0
        image_array = image_array.reshape(1, 28, 28, 1)
    else:
        image_array = image_array.flatten().reshape(1, -1) / 255.0
        image_array = pca.transform(image_array)

    return image_array

from flask import send_from_directory

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    model_type = request.form.get('model', 'cnn').lower()
    image_file = request.files['image']

    if model_type == 'cnn':
        processed = preprocess_image(image_file, for_cnn=True)
        prediction = cnn_model.predict(processed)
        predicted_digit = int(np.argmax(prediction))
        used_model = "CNN"
    elif model_type == 'svm':
        processed = preprocess_image(image_file, for_cnn=False)
        predicted_digit = int(svm_model.predict(processed)[0])
        used_model = "SVM"
    else:
        return jsonify({'error': 'Invalid model type. Use cnn or svm'}), 400

    return jsonify({
        'model_used': used_model,
        'prediction': predicted_digit
    })

