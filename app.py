import requests
import zipfile
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Set default encoding to UTF-8
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Function to print with UTF-8 encoding
def print_utf8(text):
    print(text.encode('utf-8', errors='replace').decode('utf-8'))

# URL of the dataset
url = "https://github.com/MrunaliB15/Project/raw/main/Dataset1.zip"
local_zip_path = "./Dataset1.zip"
extract_path = "./Dataset1"

# Ensure the target directory exists
os.makedirs(os.path.dirname(local_zip_path), exist_ok=True)

# Download the file
response = requests.get(url, stream=True)
with open(local_zip_path, 'wb') as file:
    for chunk in response.iter_content(chunk_size=128):
        file.write(chunk)

print_utf8(f"Downloaded dataset to {local_zip_path}")

# Extract the zip file
if os.path.exists(local_zip_path):
    with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print_utf8(f"Extracted dataset to {extract_path}")

    # Clean up: Remove the zip file
    os.remove(local_zip_path)
else:
    print_utf8(f"File {local_zip_path} does not exist.")

# Define image dimensions
image_height = 224
image_width = 224

# Create ImageDataGenerator
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
num_classes = 6

# Build the model
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
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
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
print_utf8("Model saved as 'saved_model.keras'")

# Plotting Training History
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
