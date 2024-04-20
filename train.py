import os
import numpy as np
from PIL import Image
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf
from tensorflow.keras import layers, models

# Load data
organized_folder = "organized_data"

# Define image size
image_size = (224, 224)


# Function to load and preprocess images
def load_images(image_path):
    img = Image.open(image_path).resize(image_size)
    img = np.array(img) / 255.0  # Normalize pixel values
    return img


# Function to load severity scores
def load_severity_scores(severity_file):
    with open(severity_file, "r") as f:
        severity_scores = [line.strip() for line in f]
    return severity_scores


# Function to load polygon coordinates
def load_polygon_coordinates(polygon_file):
    with open(polygon_file, "r") as f:
        polygon_coordinates = [eval(line.strip()) for line in f]
    return polygon_coordinates


# Function to preprocess data
def preprocess_data(organized_folder):
    images = []
    pixel_differences = []
    severity_labels = []
    polygon_coordinates = []

    for sample_folder in os.listdir(organized_folder):
        if not os.path.isdir(os.path.join(organized_folder, sample_folder)):
            continue

        # Load post-disaster image
        post_disaster_img = load_images(os.path.join(organized_folder, sample_folder, "post_disaster.png"))

        # Load pixel-wise difference image
        pixel_difference_img = load_images(os.path.join(organized_folder, sample_folder, "pixel_difference.png"))

        # Load severity scores
        severity_file = os.path.join(organized_folder, sample_folder, "severity_scores.txt")
        severity_scores = load_severity_scores(severity_file)

        # Load polygon coordinates
        polygon_file = os.path.join(organized_folder, sample_folder, "polygon_coordinates.txt")
        polygons = load_polygon_coordinates(polygon_file)

        # Append data for each building instance in the sample
        for severity, polygon in zip(severity_scores, polygons):
            images.append(post_disaster_img)
            pixel_differences.append(pixel_difference_img)
            severity_labels.append(severity)
            polygon_coordinates.append(polygon)

    return np.array(images), np.array(pixel_differences), severity_labels, polygon_coordinates


# Prepare input and output data
images, pixel_differences, severity_labels, polygon_coordinates = preprocess_data(organized_folder)

# Convert severity labels to numerical labels
# Convert severity labels to numerical labels
label_encoder = LabelEncoder()
severity_labels_encoded = label_encoder.fit_transform(severity_labels)

# No need to one-hot encode severity labels since we're using sparse categorical crossentropy loss
# Convert to TensorFlow tensor type

# One-hot encode severity labels
onehot_encoder = OneHotEncoder()
severity_labels_onehot = onehot_encoder.fit_transform(severity_labels_encoded.reshape(-1, 1))

# Split data into training and validation sets
X_train, X_val, pd_train, pd_val, y_train, y_val = train_test_split(images, pixel_differences, severity_labels_onehot,
                                                                    test_size=0.1, random_state=42)



y_train = tf.convert_to_tensor([severity_labels_encoded[i] for i in y_train.indices])
y_val = tf.convert_to_tensor([severity_labels_encoded[i] for i in y_val.indices])

y_train = tf.keras.utils.to_categorical(y_train, num_classes=4)
y_val = tf.keras.utils.to_categorical(y_val, num_classes=4)


# Define CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(4, activation='softmax')  # Adjust output size based on severity classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit([X_train, pd_train], y_train, epochs=30, batch_size=32, validation_data=([X_val, pd_val], y_val))

# Evaluate the model
loss, accuracy = model.evaluate([X_val, pd_val], y_val)
print(f'Validation accuracy: {accuracy}')


model.save('building_detection_model.h5')
