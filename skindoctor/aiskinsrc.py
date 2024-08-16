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
    img = img.resize((256, 256))
    
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
    model = tf.keras.models.load_model("models/skindoctor.keras")
    
    # Preprocess the image
    processed_img = preprocess_image(img)
    
    # Add batch dimension
    input_img = np.expand_dims(processed_img, 0)
    
    # Make prediction
    yhat = model.predict(input_img)

    acne = yhat[0][0] * 100
    healthy = yhat[0][1] * 100
    melanoma = yhat[0][2] * 100

    # Format the string
    result_string = f"acne: {acne:.2f}%\nhealthy: {healthy:.2f}%\nmelanoma: {melanoma:.2f}%"

    return result_string

# Example usage:
# from PIL import Image
# img = Image.open('acne_test.jpeg')
# print(predict_image_class(img))