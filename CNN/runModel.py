#runModel.py

import tensorflow as tf
import numpy as np
from PIL import Image
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def preprocess_image(img):
    """
    Preprocess the input image for the model.
    """
    # Resize the image
    img = img.resize((128, 128))
    
    # Convert to RGB if it's not
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Convert to numpy array and normalize
    img_array = np.array(img).astype(np.float32) / 255.0
    
    return img_array

def predict_image_class(img):
    """
    Predict the class probabilities of the input image using the model.

    :param img: Input image as a PIL Image object
    :return: Predicted class probabilities as a formatted string
    """
    model = tf.keras.models.load_model("../CNN/models/tuned_model.keras")
    
    # Preprocess the image
    processed_img = preprocess_image(img)
    
    # Add batch dimension
    input_img = np.expand_dims(processed_img, 0)
    
    # Make prediction
    predictions = model.predict(input_img)
    predictions = predictions[0]

        # Class labels
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


    # Map predictions to class labels and format output
    # Map predictions to class labels and format output
    output = "\n".join(
        f"{label}: {float(probability):.2%}"  # Ensure probability is a float
        for label, probability in zip(class_labels, predictions)
    )



    return output

# print(predict_image_class(Image.open("test_images/acne_test.jpeg")))
