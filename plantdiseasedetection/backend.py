import tensorflow as tf
from flask import Flask, request, jsonify
import numpy as np
import json

app = Flask(__name__)

# Load the TFLite model
model_path = "model/leaf_disease_model.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Load remedies JSON
with open(r"C:\Users\Khushi\OneDrive\Desktop\Plant Disease Detection\src\plantdiseasedetection\remedies.json", "r") as f:
    remedies = json.load(f)


def preprocess_image(image_bytes):
    # Preprocess the image (adjust this based on your model's requirements)
    image = tf.image.decode_image(image_bytes, channels=3)
    image = tf.image.resize(image, [224, 224])  # Assuming model requires 224x224
    image = tf.expand_dims(image, axis=0)
    image = image / 255.0
    return image

def predict(image_bytes):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preprocess the image
    input_tensor = preprocess_image(image_bytes)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()

    # Get prediction
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_label = np.argmax(output_data)

    return predicted_label

@app.route("/predict", methods=["POST"])
def predict_disease():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    # Get the uploaded image
    image_file = request.files["image"]
    image_bytes = image_file.read()

    # Predict disease
    predicted_label = predict(image_bytes)

    # Find remedy
    disease_name = remedies.get(str(predicted_label), {}).get("disease", "Unknown")
    remedy = remedies.get(str(predicted_label), {}).get("remedy", "No remedy available")

    return jsonify({
        "disease": disease_name,
        "remedy": remedy
    })

@app.route("/", methods=["GET"])
def test_route():
    return "Server is up and running!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
