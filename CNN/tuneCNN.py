import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras_tuner import Hyperband
import os
import scipy
import numpy as np
from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

print()
print()
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(get_available_gpus())
print()
print()

# Get the current working directory
cwd = os.getcwd()
print("Current Working Directory:", cwd)

# Step 1: Define paths
train_dir = "./Split_smol/train"
val_dir = "./Split_smol/val"

# Step 2: Preprocess the data
# Using ImageDataGenerator for image augmentation and normalization
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,      # Normalize pixel values
    rotation_range=20,      # Random rotation
    width_shift_range=0.2,  # Random horizontal shifts
    height_shift_range=0.2, # Random vertical shifts
    shear_range=0.2,        # Shearing
    zoom_range=0.2,         # Zooming
    horizontal_flip=True,   # Horizontal flipping
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Loading data from directories
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),  # Resize images to 128x128
    batch_size=32,           # Number of images per batch
    class_mode="categorical" # For multi-class classification
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode="categorical"
)

# Step 3: Define a function for building the model (for Keras Tuner)
def build_model(hp):
    model = Sequential()

    # Convolutional Layers with hyperparameter tuning for filters and dropout
    model.add(Conv2D(
        filters=hp.Choice('conv_1_filters', [32, 64, 128]),
        kernel_size=(3, 3),
        activation='relu',
        input_shape=(128, 128, 3)
    ))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(
        filters=hp.Choice('conv_2_filters', [64, 128, 256]),
        kernel_size=(3, 3),
        activation='relu'
    ))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(
        filters=hp.Choice('conv_3_filters', [128, 256, 512]),
        kernel_size=(3, 3),
        activation='relu'
    ))
    model.add(MaxPooling2D((2, 2)))

    # Flatten and Fully Connected Layers
    model.add(Flatten())
    model.add(Dense(
        units=hp.Choice('dense_units', [64, 128, 256]),
        activation='relu'
    ))
    model.add(Dropout(hp.Choice('dropout_rate', [0.3, 0.5, 0.7])))
    model.add(Dense(train_generator.num_classes, activation='softmax'))

    # Compile the model with hyperparameter tuning for learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])
        ),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Step 4: Use Keras Tuner to find the best hyperparameters
tuner = Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=20,
    factor=3,
    directory='hyperband_tuning',
    project_name='disease_classifier'
)

def tune_model():
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    # Run the hyperparameter search
    tuner.search(
        train_generator,
        validation_data=val_generator,
        epochs=20,
        callbacks=[stop_early]
    )

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("""\nThe optimal hyperparameters are:
    - Conv Layer 1 Filters: {}
    - Conv Layer 2 Filters: {}
    - Conv Layer 3 Filters: {}
    - Dense Units: {}
    - Dropout Rate: {}
    - Learning Rate: {}
    """.format(
        best_hps.get('conv_1_filters'),
        best_hps.get('conv_2_filters'),
        best_hps.get('conv_3_filters'),
        best_hps.get('dense_units'),
        best_hps.get('dropout_rate'),
        best_hps.get('learning_rate')
    ))

    return best_hps

# Tune the model
best_hps = tune_model()

# Step 5: Train the model with the best hyperparameters
model = tuner.hypermodel.build(best_hps)
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=25
)

# Step 6: Evaluate the model
loss, accuracy = model.evaluate(val_generator)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# Step 7: Save the model
model.save("disease_classifier_tuned_model.h5")
