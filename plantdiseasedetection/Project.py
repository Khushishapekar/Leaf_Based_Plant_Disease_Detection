import requests
import zipfile
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import sys

# Set default encoding to UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Function to print with UTF-8 encoding
def print_utf8(text):
    print(text.encode('utf-8', errors='replace').decode('utf-8'))

# URL of the dataset
url = "https://github.com/MrunaliB15/Project/raw/main/Dataset1.zip"
# Correct path where you have write permissions
local_zip_path = r"C:\Users\Khushi\OneDrive\Desktop\Plant Disease Detection\Dataset1.zip"

# Ensure the target directory exists
os.makedirs(os.path.dirname(local_zip_path), exist_ok=True)

# Download the file
response = requests.get(url, stream=True)
with open(local_zip_path, 'wb') as file:
    for chunk in response.iter_content(chunk_size=128):
        file.write(chunk)

print_utf8(f"Downloaded dataset to {local_zip_path}")

# Verify if the file exists before attempting to open it
if os.path.exists(local_zip_path):
    # Extract the zip file
    extract_path = r"C:\Users\Khushi\OneDrive\Desktop\Plant Disease Detection\Dataset1"
    os.makedirs(extract_path, exist_ok=True)  # Ensure the extraction directory exists
    with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print_utf8(f"Extracted dataset to {extract_path}")

    # Clean up: Remove the zip file if you want
    os.remove(local_zip_path)
else:
    print_utf8(f"File {local_zip_path} does not exist.")

# Using the correct path for the extracted dataset
extract_path = r"C:\Users\Khushi\OneDrive\Desktop\Plant Disease Detection\Dataset1"

# Update image dimensions
image_height = 224  # Set desired image height
image_width = 224   # Set desired image width

# Create ImageDataGenerator with updated target size
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = train_datagen.flow_from_directory(
    extract_path,
    target_size=(image_height, image_width),
    batch_size=32,
    class_mode='sparse',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    extract_path,
    target_size=(image_height, image_width),
    batch_size=32,
    class_mode='sparse',
    subset='validation'
)

# Define the number of classes
num_classes = 6  # Updated to match the actual number of classes

# Adjust the model architecture
model = models.Sequential([
    layers.Input(shape=(image_height, image_width, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')  # Updated output layer
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Model training
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=10
)

# Create the directory if it doesn't exist
if not os.path.exists('model'):
    os.makedirs('model')

# Save the model
model.save('model/saved_model.keras')

# Plotting Training History
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

# Real-time Plant Disease Detection
# Load the trained model
model = tf.keras.models.load_model('model/saved_model.keras')

# Class labels (assuming there are six classes: healthy, disease_1, disease_2, etc.)
class_labels = ['healthy', 'disease_1', 'disease_2', 'disease_3', 'disease_4', 'disease_5']

# Function to preprocess the image
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))  # Resize to match model input
    image = np.expand_dims(image, axis=0)
    return image / 255.0

# Capture video from the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    processed_frame = preprocess_image(frame)

    # Predict the disease
    predictions = model.predict(processed_frame)
    disease = class_labels[np.argmax(predictions, axis=1)[0]]

    # Display the result
    cv2.putText(frame, f'Disease: {disease}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('Plant Disease Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Streamlit App
# Load your trained model (use either the Keras model or a TFLite model)
model = tf.keras.models.load_model('model/saved_model.keras')

# Define class labels
class_labels = ['healthy', 'disease_1', 'disease_2', 'disease_3', 'disease_4', 'disease_5']

# Function to preprocess image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to match model input
    image = np.array(image)
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def main():
    st.title("Plant Disease Detection")
    st.write("Upload an image of a plant leaf to detect its health status.")

    file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if file is not None:
        image = Image.open(file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        processed_image = preprocess_image(image)

        # Make prediction
        prediction = model.predict(processed_image)
        disease = class_labels[np.argmax(prediction, axis=1)[0]]

        st.write(f'The leaf is classified as: {disease}')

if __name__ == '__main__':
    main()


# C:\Users\Khushi\OneDrive\Desktop\Mrunali's Project\PlantDiseaseApp\leafdetection>