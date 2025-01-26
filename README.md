# Skin Disease Classification and Diagnosis Project

## Overview
This project is an AI-powered application designed to assist in diagnosing and recommending treatments for various skin conditions. The project leverages a convolutional neural network (CNN) for image-based skin disease classification and integrates an interactive chatbot for symptom-based diagnosis and treatment recommendations.

---

## Sample Usage


```plaintext
===============================================
This app will try to help you with skin problems. Let's get started. Do you want to talk about any skin issues you're facing? (Y/N)

You: y
===============================================
Have you been diagnosed for this skin issue?

You: n
===============================================
Describe your symptoms

You: red and dry skin on elbows
Possible diagnoses could include eczema, psoriasis, allergic contact dermatitis, or dry skin (xerosis). A medical professional would need to perform a physical examination and possibly further tests to confirm the diagnosis.
===============================================
Please upload the file path of an image of your skin condition

You: "C:\Users\samne\Downloads\psoriasis-pic.jpg"

Image successfully loaded. Size: (1248, 832)
1/1 [==============================] - 6s 6s/step

Your image results are 
Actinic keratosis: 2.30%
Atopic Dermatitis: 50.94%
Benign keratosis: 2.20%
Dermatofibroma: 1.88%
Melanocytic nevus: 0.19%
Melanoma: 1.89%
Squamous cell carcinoma: 2.99%
Tinea Ringworm: 37.54%
Vascular lesion: 0.07%
===============================================
Would you like to get recommendations for a skin disease? Type 'N' or type the skin disease you would like to ask about.

You: psoriasis

1. Use gentle, fragrance-free skincare products to avoid irritating your skin.
2. Moisturize regularly to help relieve dryness and reduce itching.
3. Consider topical treatments or medications prescribed by a dermatologist.
4. Manage stress, as it can exacerbate psoriasis symptoms.
===============================================
Continue chatting or press Q to move on

You: q

Thanks for using derm AI

Here is your diagnosis:

Possible diagnoses could include eczema, psoriasis, allergic contact dermatitis, or dry skin (xerosis). A medical professional would need to perform a physical examination and possibly further tests to confirm the diagnosis.

Here are your recommendations:

Prescriptive Summary:
1. Use gentle, fragrance-free skincare products.
2. Moisturize regularly to reduce dryness and itching.
3. Consult a dermatologist for topical treatments or medications.
4. Manage stress to alleviate psoriasis symptoms.
```

---

## Key Features
1. **Image Classification**: Accurately identifies skin diseases from uploaded images.
2. **Symptom-Based Diagnosis**: Generates potential diagnoses based on user-described symptoms.
3. **AI-Powered Recommendations**: Provides treatment suggestions for diagnosed conditions.
4. **Interactive Chatbot**: Engages users through a conversational interface to enhance user experience.

---

## Dataset Information
**Source**: [Skin Disease Classification Image Dataset](https://www.kaggle.com/datasets/riyaelizashaju/skin-disease-classification-image-dataset)

### Dataset Details
- Categorized images of skin conditions, including:
  - Actinic keratosis
  - Atopic Dermatitis
  - Benign keratosis
  - Dermatofibroma
  - Melanocytic nevus
  - Melanoma
  - Squamous cell carcinoma
  - Tinea Ringworm Candidiasis
  - Vascular lesion
- Split into training and validation sets for model training and evaluation.

---

## Workflow

### **1. Preprocessing**
- Images are resized to **128x128** pixels.
- Data augmentation applied (rotation, zooming, shearing, and flipping).
- Pixel values are normalized to the range **[0, 1]**.

### **2. Model Architecture**
- **Convolutional Layers**: Extract spatial features.
- **MaxPooling Layers**: Reduce feature map dimensions.
- **Dense Layers**: Capture non-linear relationships.
- **Dropout**: Prevent overfitting.

### **3. Hyperparameter Tuning**
- Tuned with **Keras Tuner's Hyperband** to optimize:
  - Number of filters in convolutional layers
  - Units in dense layers
  - Dropout rates
  - Learning rates

### **4. Training**
- Model is trained for **25 epochs** with **early stopping** to prevent overfitting.
- Validation data used to evaluate performance.

### **5. Deployment and Chatbot Integration**
- Deployed as an interactive chatbot:
  - Accepts image uploads for classification.
  - Accepts symptom descriptions for diagnosis.
  - Provides recommendations for the diagnosed conditions.

---

## How to Run the Project

### **1. Prerequisites**
- Python 3.10 or higher
- OpenAI API Key: **Set the key as an environment variable**:
  
```bash
export OPENAI_API_KEY=your_openai_key
```

### **2. Setup**
1. **Install Required Libraries**:
   ```bash
   pip install tensorflow keras-tuner Pillow openai
   ```
2. **Download the Dataset**:
   - Visit [Kaggle](https://www.kaggle.com/datasets/riyaelizashaju/skin-disease-classification-image-dataset).
   - Download the dataset and place it in the project folder under `./data/train` and `./data/val`.

3. **Run the Training Script**:
   ```bash
   cd CNN
   python tuneCNN.py
   ```

4. **Run the Chatbot Application**:
   ```bash
   cd app
   python main.py
   ```

---



## Credits
- **Dataset**: [Riya Eliza Shaju on Kaggle](https://www.kaggle.com/datasets/riyaelizashaju/skin-disease-classification-image-dataset)
- **Libraries Used**: TensorFlow, Keras, Keras Tuner, PIL, OpenAI

---

## Notes
- Always consult a medical professional for an accurate diagnosis and treatment plan. This application is designed for educational purposes and not as a substitute for professional medical advice.

---