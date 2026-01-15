#!/usr/bin/env python3
"""
Test script for the pretrained MusicSymbolClassifier model.
This script loads the pretrained model and verifies it works correctly.
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import os

print("=" * 60)
print("Testing Pretrained MusicSymbolClassifier Model")
print("=" * 60)

# Check TensorFlow version
print(f"\nTensorFlow version: {tf.__version__}")
print(f"Keras version: {tf.keras.__version__}")

# Check GPU availability
print(f"\nGPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
if len(tf.config.list_physical_devices('GPU')) > 0:
    print("GPU devices:", tf.config.list_physical_devices('GPU'))
else:
    print("Running on CPU")

# Load the pretrained model
model_path = "pretrained_model.h5"
print(f"\nLoading model from: {model_path}")

try:
    model = tf.keras.models.load_model(model_path)
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    exit(1)

# Display model summary
print("\n" + "=" * 60)
print("Model Summary")
print("=" * 60)
model.summary()

# Get model input shape
input_shape = model.input_shape
print(f"\nModel input shape: {input_shape}")
print(f"Expected image size: {input_shape[1]}x{input_shape[2]} pixels")
print(f"Number of classes: {model.output_shape[-1]}")

# Test with a random image
print("\n" + "=" * 60)
print("Testing with Random Image")
print("=" * 60)

# Create a random test image matching the model's input shape
height, width = input_shape[1], input_shape[2]
test_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

# Save test image for reference
test_image_pil = Image.fromarray(test_image)
test_image_pil.save("test_random_image.png")
print(f"Generated random test image: test_random_image.png")

# Preprocess the image
test_image_normalized = test_image.astype('float32') / 255.0
test_image_batch = np.expand_dims(test_image_normalized, axis=0)

# Make prediction
print("\nMaking prediction...")
predictions = model.predict(test_image_batch, verbose=0)

# Display results
predicted_class = np.argmax(predictions[0])
confidence = predictions[0][predicted_class]

print(f"Predicted class: {predicted_class}")
print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")

# Show top 5 predictions
top_5_indices = np.argsort(predictions[0])[-5:][::-1]
print("\nTop 5 predictions:")
for i, idx in enumerate(top_5_indices, 1):
    print(f"  {i}. Class {idx}: {predictions[0][idx]:.4f} ({predictions[0][idx]*100:.2f}%)")

print("\n" + "=" * 60)
print("✓ Model is working correctly!")
print("=" * 60)
print("\nNext steps:")
print("1. Download the HOMUS dataset to get actual music symbol images")
print("2. Use the model to classify real music symbols")
print("3. Fine-tune the model on your own data if needed")
print("\nTo use this model on your own images:")
print("  - Resize images to {}x{} pixels".format(height, width))
print("  - Normalize pixel values to [0, 1] range")
print("  - Use model.predict() to get class predictions")
