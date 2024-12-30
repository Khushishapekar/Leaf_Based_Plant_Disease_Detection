import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.filechooser import FileChooserIconView
import tensorflow as tf
import cv2
import numpy as np
import os
import sys

# Set default encoding to UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Function to print with UTF-8 encoding
def print_utf8(text):
    print(text.encode('utf-8', errors='replace').decode('utf-8'))

class LeafDetectorApp(App):
    def build(self):
        # Ensure the import and path setup
        import os
        
        # Print the current working directory
        print_utf8(f"Current working directory: {os.getcwd()}")

        # Define the model path using an absolute path
        model_path = os.path.join(os.getcwd(), 'model', 'saved_model.keras')
        print_utf8(f"Model path: {model_path}")

        # Check if the model file exists
        if not os.path.exists(model_path):
            print_utf8(f"Model file not found at: {model_path}")
            # Optionally, create a placeholder UI element to indicate the error
            layout = BoxLayout(orientation='vertical')
            self.label = Label(text=f"Model file not found at: {model_path}")
            layout.add_widget(self.label)
            return layout

        self.model = tf.keras.models.load_model(model_path)
        self.class_labels = ['healthy', 'disease_1', 'disease_2', 'disease_3', 'disease_4', 'disease_5']

        layout = BoxLayout(orientation='vertical')
        self.image = Image(source='assets/some_image.jpg')
        self.label = Label(text="Upload an image of a plant leaf to detect its health status.")
        self.filechooser = FileChooserIconView()  # Use custom file chooser if needed
        self.button = Button(text="Classify Image")
        self.button.bind(on_press=self.classify_image)

        layout.add_widget(self.label)
        layout.add_widget(self.image)
        layout.add_widget(self.filechooser)
        layout.add_widget(self.button)

        return layout

    def classify_image(self, instance):
        selected = self.filechooser.selection
        if not selected:
            self.label.text = "No image selected!"
            return
        
        image_path = selected[0]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = np.expand_dims(image, axis=0) / 255.0

        # Predict the disease
        predictions = self.model.predict(image)
        predicted_class = self.class_labels[np.argmax(predictions, axis=1)[0]]

        self.label.text = f'The leaf is classified as: {predicted_class}'
        self.image.source = image_path

if __name__ == '__main__':
    LeafDetectorApp().run()
