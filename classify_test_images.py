#!/usr/bin/env python3
"""
Classify music symbol images from the testingarea folder.
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import os
import glob

# Class names for the 79 music symbol classes (from HOMUS dataset)
CLASS_NAMES = [
    "12-8-Time", "2-2-Time", "2-4-Time", "3-4-Time", "3-8-Time", "4-4-Time",
    "6-8-Time", "9-8-Time", "Accent", "Barline", "Bass-Clef", "Beam",
    "Dot", "Double-Sharp", "Eighth-Note", "Eighth-Rest", "Flat", "Grace-Note-Acciaccatura",
    "Grace-Note-Appoggiatura", "Half-Note", "Ledger-Line", "Natural", "Quarter-Note",
    "Quarter-Rest", "Repeat", "Sharp", "Sixteenth-Note", "Sixteenth-Rest",
    "Sixty-Four-Note", "Sixty-Four-Rest", "Slur", "Staff-Line", "Stem",
    "Tie", "Time-Signature", "Treble-Clef", "Triplet", "Tuplet",
    "Whole-Half-Rest", "Whole-Note", "C-Clef", "Common-Time", "Cut-Time",
    "Coda", "Fermata", "Pedal-Down", "Pedal-Up", "Segno", "Staccato",
    "Tenuto", "Trill", "Turn", "Arpeggio", "Brace", "Bracket",
    "Caesura", "Chord", "Cresc", "Decresc", "Dynamics-f", "Dynamics-ff",
    "Dynamics-fff", "Dynamics-mf", "Dynamics-mp", "Dynamics-p", "Dynamics-pp",
    "Dynamics-ppp", "Glissando", "Mordent", "Multiple-Rest", "Ottava",
    "Ottava-Alta", "Ottava-Bassa", "Pedal", "Rehearsal-Mark", "Repeat-Dots",
    "Rest-Whole", "Rest-Half", "Slur-Down", "Slur-Up"
]

def preprocess_image(image_path, target_size=(96, 96)):
    """Load and preprocess an image for the model."""
    # Load image
    image = Image.open(image_path)

    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Resize to model input size
    image = image.resize(target_size)

    # Convert to numpy array and normalize
    image_array = np.array(image).astype('float32') / 255.0

    # Add batch dimension
    image_batch = np.expand_dims(image_array, axis=0)

    return image_batch, image

def classify_image(model, image_path, class_names):
    """Classify a single image and return results."""
    # Preprocess image
    image_batch, original_image = preprocess_image(image_path)

    # Make prediction
    predictions = model.predict(image_batch, verbose=0)

    # Get top 5 predictions
    top_5_indices = np.argsort(predictions[0])[-5:][::-1]

    results = []
    for idx in top_5_indices:
        confidence = predictions[0][idx]
        class_name = class_names[idx] if idx < len(class_names) else f"Class {idx}"
        results.append({
            'class_id': int(idx),
            'class_name': class_name,
            'confidence': float(confidence)
        })

    return results

def main():
    print("=" * 80)
    print("Music Symbol Classifier - Testing on Real Images")
    print("=" * 80)

    # Load the pretrained model
    model_path = "pretrained_model.h5"
    print(f"\nLoading model: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print("✓ Model loaded successfully!")

    # Get test images
    testingarea_path = "../testingarea/"
    image_files = sorted(glob.glob(os.path.join(testingarea_path, "*.png")))

    if not image_files:
        print(f"\n✗ No images found in {testingarea_path}")
        return

    print(f"\nFound {len(image_files)} images to classify\n")

    # Classify each image
    for i, image_path in enumerate(image_files, 1):
        filename = os.path.basename(image_path)
        print("=" * 80)
        print(f"Image {i}/{len(image_files)}: {filename}")
        print("-" * 80)

        try:
            results = classify_image(model, image_path, CLASS_NAMES)

            # Display results
            top_prediction = results[0]
            print(f"\n🎵 PREDICTION: {top_prediction['class_name']}")
            print(f"   Confidence: {top_prediction['confidence']:.2%}")

            print("\n   Top 5 predictions:")
            for j, result in enumerate(results, 1):
                bar_length = int(result['confidence'] * 40)
                bar = "█" * bar_length + "░" * (40 - bar_length)
                print(f"   {j}. {result['class_name']:<30} {bar} {result['confidence']:6.2%}")

            print()

        except Exception as e:
            print(f"✗ Error classifying image: {e}\n")

    print("=" * 80)
    print("✓ Classification complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
