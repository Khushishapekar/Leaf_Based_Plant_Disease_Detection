import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model('model/saved_model.keras')

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open('model/leaf_disease_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("TFLite model saved as 'leaf_disease_model.tflite'")
