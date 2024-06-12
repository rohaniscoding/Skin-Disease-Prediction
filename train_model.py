import os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras import layers, models

# Update file paths
metadata_path = r"C:\Users\DELL\Desktop\Skin_Cancer_Prediction_Project\dataset\HAM10000_metadata.csv"
images_folder_path = r"C:\Users\DELL\Desktop\Skin_Cancer_Prediction_Project\images"

# Load metadata
metadata = pd.read_csv(metadata_path)

# Define symptoms for each class
symptoms = {
    'akiec': 'Actinic keratoses and intraepithelial carcinoma / Bowen\'s disease',
    'bcc': 'Basal cell carcinoma',
    'bkl': 'Benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses)',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic nevi',
    'vasc': 'Vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage)'
}

# Print symptoms for each class
print("Symptoms for each class:")
for label, symptom in symptoms.items():
    print(f"{label}: {symptom}")

# Load images
images = []
labels = []

for index, row in metadata.iterrows():
    image_path = os.path.join(images_folder_path, row['image_id'] + '.jpg')
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))  # Resize images to 128x128
    images.append(img)
    labels.append(row['dx'])

# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Encode labels
encoder = LabelEncoder()
labels_encoded = encoder.fit_transform(labels)
labels_onehot = to_categorical(labels_encoded)

# Split data into training, validation, and test sets
train_images, test_images, train_labels, test_labels = train_test_split(images, labels_onehot, test_size=0.2, random_state=42)
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# Normalize pixel values
train_images = train_images / 255.0
val_images = val_images / 255.0
test_images = test_images / 255.0

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
    layers.Dropout(0.5),
    layers.Dense(7, activation='softmax')  # Assuming there are 7 classes
])

# Print model summary
model.summary()

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(val_images, val_labels))

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_images, test_labels)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)