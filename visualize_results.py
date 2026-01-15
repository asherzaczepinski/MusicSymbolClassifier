#!/usr/bin/env python3
"""
Visualize the classification results with side-by-side comparisons.
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Class names
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
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image_resized = image.resize(target_size)
    image_array = np.array(image_resized).astype('float32') / 255.0
    image_batch = np.expand_dims(image_array, axis=0)
    return image_batch, image, image_resized

def main():
    print("Creating visualization of classification results...")

    # Load model
    model = tf.keras.models.load_model("pretrained_model.h5")

    # Get test images
    testingarea_path = "../testingarea/"
    image_files = sorted(glob.glob(os.path.join(testingarea_path, "*.png")))

    # Create figure
    n_images = len(image_files)
    fig = plt.figure(figsize=(20, 5 * n_images))
    gs = gridspec.GridSpec(n_images, 3, width_ratios=[1, 1, 2], hspace=0.3, wspace=0.3)

    for i, image_path in enumerate(image_files):
        filename = os.path.basename(image_path)

        # Preprocess and classify
        image_batch, original_image, resized_image = preprocess_image(image_path)
        predictions = model.predict(image_batch, verbose=0)

        # Get top 5 predictions
        top_5_indices = np.argsort(predictions[0])[-5:][::-1]
        top_5_names = [CLASS_NAMES[idx] for idx in top_5_indices]
        top_5_confidences = [predictions[0][idx] for idx in top_5_indices]

        # Plot original image
        ax1 = fig.add_subplot(gs[i, 0])
        ax1.imshow(original_image, cmap='gray')
        ax1.set_title(f'Original: {filename}', fontsize=10)
        ax1.axis('off')

        # Plot resized image (what model sees)
        ax2 = fig.add_subplot(gs[i, 1])
        ax2.imshow(resized_image)
        ax2.set_title('Resized to 96x96\n(Model Input)', fontsize=10)
        ax2.axis('off')

        # Plot predictions
        ax3 = fig.add_subplot(gs[i, 2])
        y_pos = np.arange(len(top_5_names))
        colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(top_5_names))]
        bars = ax3.barh(y_pos, top_5_confidences, color=colors)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(top_5_names, fontsize=10)
        ax3.set_xlabel('Confidence', fontsize=10)
        ax3.set_title(f'Top Prediction: {top_5_names[0]} ({top_5_confidences[0]:.1%})',
                     fontsize=12, fontweight='bold')
        ax3.set_xlim([0, 1])
        ax3.grid(axis='x', alpha=0.3)

        # Add percentage labels on bars
        for j, (bar, conf) in enumerate(zip(bars, top_5_confidences)):
            ax3.text(conf + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{conf:.1%}', va='center', fontsize=9)

    plt.suptitle('Music Symbol Classification Results', fontsize=16, fontweight='bold', y=0.995)

    # Save figure
    output_path = 'classification_results_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to: {output_path}")

    plt.close()

if __name__ == "__main__":
    main()
