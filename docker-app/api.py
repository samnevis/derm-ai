#docker-app/api.py
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import boto3
import os

app = Flask(__name__)

# S3 Config
BUCKET_NAME = "derm-ai"  # Your actual S3 bucket name
MODEL_FILE = "tuned_model.keras"  # Your actual model filename


# Disable TensorFlow optimizations for compatibility
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Download model from S3 (if not present)
if not os.path.exists(MODEL_FILE):
    s3 = boto3.client('s3')
    s3.download_file(BUCKET_NAME, f"models/{MODEL_FILE}", MODEL_FILE)

# Load TensorFlow model
model = tf.keras.models.load_model(MODEL_FILE)

# Class labels for predictions
class_labels = [
    "Actinic keratosis",
    "Atopic Dermatitis",
    "Benign keratosis",
    "Dermatofibroma",
    "Melanocytic nevus",
    "Melanoma",
    "Squamous cell carcinoma",
    "Tinea Ringworm",
    "Vascular lesion"
]

def preprocess_image(img):
    """
    Preprocess the input image for the model.
    """
    img = img.resize((128, 128))  # Resize to match model input
    
    if img.mode != 'RGB':
        img = img.convert('RGB')  # Convert to RGB if needed
    
    img_array = np.array(img).astype(np.float32) / 255.0  # Normalize
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get base64 image from JSON request
        data = request.json['image']
        
        # Decode the image
        image_bytes = base64.b64decode(data)
        image = Image.open(io.BytesIO(image_bytes))

        # Preprocess the image
        processed_img = preprocess_image(image)

        # Add batch dimension
        input_img = np.expand_dims(processed_img, axis=0)

        # Make prediction
        predictions = model.predict(input_img)[0]

        # Format predictions
        output = {
            label: f"{float(probability):.2%}"
            for label, probability in zip(class_labels, predictions)
        }

        return jsonify(output)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
