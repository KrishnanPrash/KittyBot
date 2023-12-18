import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image
import pickle

# Function to load and preprocess images
def load_and_preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((128, 128))  # Resize image to your preferred size
    img_array = np.array(img) / 255.0  # Normalize pixel values to between 0 and 1
    return img_array

# Function to load data from directories
def load_data(directory):
    data = []
    labels = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if '.jpg' != file[-4:]: continue
            file_path = os.path.join(root, file)
            label = os.path.basename(root)
            data.append(load_and_preprocess_image(file_path))
            labels.append(label)

    return np.array(data), np.array(labels)

# Load data from the specified folders
train_data_cinder, train_labels_cinder = load_data("cinder")
train_data_stripe, train_labels_stripe = load_data("stripe")
train_data_both, train_labels_both = load_data("both")

# Combine data from all folders
train_data = np.concatenate([train_data_cinder, train_data_stripe, train_data_both])
train_labels = np.concatenate([train_labels_cinder, train_labels_stripe, train_labels_both])

# Shuffle the data
indices = np.arange(train_data.shape[0])
np.random.shuffle(indices)
train_data = train_data[indices]
train_labels = train_labels[indices]

# Define the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(4, activation='softmax')  # 4 output classes: cinder, stripe, both, none
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_data, train_labels, epochs=10, validation_split=0.2)

# Save the model accuracy using pickle
accuracy = model.evaluate(train_data, train_labels)[1]
with open('model_accuracy.pkl', 'wb') as f:
    pickle.dump(accuracy, f)

# Print the model accuracy
print(f'Model Accuracy: {accuracy}')

# Save the trained model
model.save('image_classifier_model.h5')