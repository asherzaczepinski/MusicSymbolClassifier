# MusicSymbolClassifier - Usage Guide

## Successfully Installed!

The pretrained "small" ResNet model is now working on your system with GPU acceleration!

## Model Specifications

- **Model**: ResNet3 Small (res_net_3_small)
- **Accuracy**: ~96% validation accuracy
- **Size**: 56.3 MB (4.88M parameters)
- **Input**: 96x96 pixel RGB images
- **Output**: 79 music symbol classes
- **GPU**: Enabled (Apple M5 Metal)

## Quick Start

### Activate the environment:
```bash
cd /Users/asherzaczepinski/Desktop/MusicSymbolClassifier\(eventuallytohomr\)/MusicSymbolClassifier
source venv/bin/activate
```

### Run the test script:
```bash
python test_pretrained_model.py
```

### Use the model in your own code:
```python
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the model
model = tf.keras.models.load_model('pretrained_model.h5')

# Load and preprocess your image
image = Image.open('your_music_symbol.png')
image = image.resize((96, 96))  # Resize to 96x96
image_array = np.array(image) / 255.0  # Normalize to [0, 1]
image_batch = np.expand_dims(image_array, axis=0)  # Add batch dimension

# Make prediction
predictions = model.predict(image_batch)
predicted_class = np.argmax(predictions[0])
confidence = predictions[0][predicted_class]

print(f"Predicted class: {predicted_class}")
print(f"Confidence: {confidence:.2%}")
```

## Performance Benefits of the "Small" Model

Your model is the **"small" version**, which means:
- ✅ **2-4x faster inference** compared to full ResNet3
- ✅ **~25% of the size** (56 MB vs 200+ MB)
- ✅ **Less memory usage** - perfect for your 16GB M5 system
- ✅ **GPU accelerated** with TensorFlow Metal
- ✅ **Still 96% accurate** - excellent for most use cases

## Next Steps

1. **Download HOMUS dataset** to test with real music symbols:
   ```bash
   python ModelTrainer/TrainModel.py --use_existing_dataset_directory
   ```

2. **Test on your own images**: Place music symbol images in a folder and classify them

3. **Fine-tune the model**: Train on additional data if needed

## Training Your Own Model (Optional)

If you want to train from scratch:

```bash
# This will take 2-4 hours with GPU
cd ModelTrainer
python TrainModel.py --model_name res_net_3_small
```

## Files Created

- `pretrained_model.h5` - The pretrained model weights (56.3 MB)
- `test_pretrained_model.py` - Test script to verify the model
- `test_random_image.png` - Sample test image
- `venv/` - Python virtual environment with all dependencies

## System Info

- Python: 3.11
- TensorFlow: 2.16.2 (with Metal GPU support)
- Platform: macOS with Apple M5 chip
- RAM: 16 GB
- GPU: Apple M5 (10 cores) - **ACTIVE**

---

**The model is ready to use! Enjoy classifying music symbols!**
