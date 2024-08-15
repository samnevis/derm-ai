import keras
import tensorflow as tf
import numpy as np
import cv2
import os


def predict_image_class(image_path):
    """
    Predict the class probabilities of the input image using the model.

    :param image_path: Path to the image file
    :return: Predicted class probabilities
    """
    #os.chdir('skindoctor')
    print(os.listdir())
    print(os.path.exists('models/skindoctor.keras'))

    model = tf.keras.models.load_model('models/skindoctor.keras')

    img = cv2.imread(image_path)
    resize = tf.image.resize(img, (256,256))

    yhat = model.predict(np.expand_dims(resize/255, 0))

    return yhat


if __name__ == "__main__":
    print(keras.version())
    test = predict_image_class('acne_test.jpeg')
    print(test)
