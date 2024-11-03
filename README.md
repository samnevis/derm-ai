# Derm AI Application

## Overview

Derm AI is an interactive application designed to help users diagnose and get recommendations on skin conditions. It leverages a combination of AI-driven text interactions through OpenAI's GPT-3.5-turbo model and a TensorFlow-based image classification model. The goal is to assist users in diagnosing common skin conditions like acne, melanoma, and healthy skin, while also providing symptom-based diagnosis and recommendations.

The application is written in Python and includes the following functionalities:
1. Conversational diagnosis and recommendations for skin issues via AI.
2. Image classification of skin conditions using a pre-trained model.
3. A user-friendly command-line interface for interaction.

## Features

- **Symptom-Based Diagnosis**: Users can describe their symptoms, and the AI will generate potential diagnoses based on their description.
- **Image-Based Classification**: Users can upload an image of their skin condition, and the application will use a TensorFlow model to classify the condition as either acne, healthy skin, or melanoma.
- **AI-Generated Recommendations**: Based on the diagnosis or the user's request, the AI can provide recommendations on skin care or potential treatments.
- **Chat-Style Interaction**: The app mimics a conversation with a doctor, asking relevant questions and guiding the user through the diagnostic and recommendation process.

## Dataset Information

To enhance the diagnostic accuracy of Derm AI, high-quality image datasets were used for training and fine-tuning the image classification model. These datasets provide a variety of labeled images of skin conditions, allowing the TensorFlow model to learn the distinguishing features of conditions like acne, healthy skin, and melanoma. Below are details on the datasets used:

### 1. [Skin Diseases Image Dataset by Ismail Promus](https://www.kaggle.com/datasets/ismailpromus/skin-diseases-image-dataset/data)

This dataset contains a diverse collection of images covering several skin diseases, providing a foundation for recognizing various dermatological conditions. It includes categories such as acne, eczema, psoriasis, and more, allowing for the potential to expand Derm AI's diagnostic range in the future. Key features of the dataset include:

- **Number of Images**: Approximately 5,000 images.
- **Image Resolution**: Moderate resolution, suitable for model training and processing.
- **Labeling**: Each image is labeled with the associated skin condition, enabling supervised training.
- **Dataset Purpose**: Primarily used for detecting and distinguishing various types of skin diseases, with an emphasis on common conditions like acne and eczema.

This dataset serves as the primary source for recognizing acne, eczema, and other non-malignant skin conditions in the current version of Derm AI.

### 2. [Melanoma Skin Cancer Dataset by Hasnain Javed](https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images)

This dataset is tailored for melanoma detection, containing thousands of images with a focus on malignant and benign skin lesions. It aids in enhancing the model's ability to identify melanoma, a critical function of the application. The dataset specifics include:

- **Number of Images**: Around 10,000 images.
- **Image Resolution**: High-quality images that allow for detailed analysis of skin lesions.
- **Labeling**: Each image is labeled as either melanoma or benign, providing a clear distinction for binary classification tasks.
- **Dataset Purpose**: Specifically used for detecting melanoma, with labeled images that differentiate malignant melanoma from benign conditions.

This dataset enables Derm AI to accurately classify images with potential melanoma, helping users identify potentially dangerous skin conditions early.

### Dataset Usage in Model Training

The datasets were preprocessed to ensure uniformity in image size, color format (RGB), and normalization, preparing them for input into the TensorFlow model. During training, data augmentation techniques such as rotation, zoom, and flip were applied to increase the model's robustness and generalizability. By combining images from both datasets, Derm AI achieves a balanced performance across common skin conditions and specific malignant cases like melanoma.

## Requirements

- **Python 3.7+**
- **TensorFlow 2.6.0+**
- **OpenAI Python client**
- **Pillow (PIL) for image processing**

To install the required packages, run the following command:

```bash
pip install tensorflow openai pillow
```

## How It Works

### Image Classification
The TensorFlow model (`skindoctor.keras`) is loaded, and an image of the user's skin condition is analyzed. The image is preprocessed (resized, converted to RGB, and normalized) before being input to the model. The model then predicts the probabilities of three possible outcomes: acne, healthy skin, or melanoma. The results are returned to the user in a human-readable format.

### Conversational Interface
The user is prompted to interact with the AI via a chat interface. The conversation follows these general steps:
1. **Symptom Inquiry**: The app asks the user if they want to discuss a skin issue and whether they have been diagnosed.
2. **Symptom Diagnosis**: If the user hasn’t been diagnosed, they are prompted to describe their symptoms, which are then used to generate potential diagnoses using the GPT-3.5-turbo model.
3. **Image Input**: The user can upload an image of their skin condition, which is classified using the TensorFlow model.
4. **Recommendations**: The AI can offer personalized recommendations based on the user's skin condition or a specific disease they inquire about.

### Example Workflow

- User starts the conversation: "Y" to discuss skin issues.
- User inputs their symptoms: "I have red itchy spots on my face."
- AI generates potential diagnoses.
- User uploads an image for analysis: `C:/path/to/skin_image.jpg`.
- AI classifies the image and returns the prediction.
- User asks for recommendations for a particular disease: "acne".
- AI provides treatment or care recommendations.
- The conversation can continue until the user decides to quit.

## Directory Structure

```plaintext
├── models/
│   └── skindoctor.keras   # Pre-trained TensorFlow model for skin classification
├── runModel.py            # Script for image preprocessing and classification
└── main.py                # Main script for user interaction and AI chat
```

## Files Explanation

### `runModel.py`

- **`preprocess_image(img)`**: Preprocesses the input image (resizing, RGB conversion, normalization).
- **`predict_image_class(img)`**: Uses the TensorFlow model to classify the image into one of the three categories: acne, healthy skin, or melanoma.

### `main.py`

- **`chat()`**: Main function that handles the interaction with the user. It starts the conversation, processes symptoms, loads images for analysis, generates recommendations, and finally, provides a summary.
- **`handle_picture(file_path)`**: Handles the user's uploaded image, processes it, and provides classification results.
- **`get_ai_response_from_msglist()`**: Sends messages to OpenAI's GPT-3.5-turbo model to generate responses based on user input.

## Installation and Setup

1. **Clone the Repository**

   Clone the repository to your local machine:

   ```bash
   git clone https://github.com/samnevis/derm-ai
   ```

2. **Install Dependencies**

   Install the required dependencies

3. **Model Setup**

   Ensure that the `skindoctor.keras` TensorFlow model is placed in the `models/` directory.

4. **Running the Application**

   To run the application, execute the following command:

   ```bash
   python main.py
   ```

   Follow the prompts in the command line to interact with the AI.

## Configuration

Ensure that you have the OpenAI API key available in your environment. This can be done by setting it as an environment variable:

```bash
export OPENAI_API_KEY='your_openai_api_key_here'
```

Alternatively, you can modify the `main.py` to load the API key directly within the script.

## Future Improvements

- **Expand Diagnostic Categories**: Add more conditions beyond acne and melanoma to the classification model.
- **GUI Interface**: Implement a graphical user interface (GUI) for easier interaction.
- **Model Optimization**: Enhance the accuracy of the classification model with more training data and better architecture.
- **Custom Recommendations**: Offer more personalized recommendations based on the user's skin type and medical history.

## Conclusion

Derm AI provides an initial step towards digital dermatology solutions, blending AI chat models with image classification techniques. It assists users in identifying common skin conditions and provides recommendations based on AI-generated insights. Although it's not a replacement for a medical professional, it offers a convenient tool for preliminary analysis and guidance.