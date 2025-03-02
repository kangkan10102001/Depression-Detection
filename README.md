# Student Depression Detection System

## Overview
The **Student Depression Detection System** is a deep learning-based project designed to detect signs of depression in students using facial images. The system employs a **Convolutional Neural Network (CNN)** for binary classification of images, distinguishing between depressive and non-depressive facial expressions. The model is trained on a dataset of facial images with data augmentation techniques to improve accuracy and generalization.

## Features
- **Preprocessing**: Automatic removal of invalid image files.
- **Image Augmentation**: Rescaling and validation split.
- **Deep Learning Model**: CNN architecture for binary classification.
- **Early Stopping**: Stops training when accuracy reaches 99%.
- **Efficient Training**: Uses TensorFlow and Keras for model development.

## Prerequisites
Ensure you have the following dependencies installed:
- Python 3.7+
- TensorFlow
- Keras
- NumPy
- OpenCV
- Pillow
- scikit-learn

You can install the dependencies using:
```bash
pip install tensorflow keras numpy opencv-python pillow scikit-learn
```

## Dataset Preparation
Place your dataset in the directory specified in the code. Ensure the dataset follows this structure:
```
D:\Depress\
    ├── Depressed\
    ├── Non-Depressed\
```
Each subfolder should contain the respective images.

## Running the Project
1. **Import required libraries:**
```python
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback
from PIL import Image
import cv2
```

2. **Preprocess the dataset:**
```python
image_exts = {'jpeg', 'jpg', 'bmp', 'png'}
for image_class in os.listdir("D:\\Depress"):
    class_path = os.path.join("D:\\Depress", image_class)
    for image in os.listdir(class_path):
        image_path = os.path.join(class_path, image)
        try:
            img = Image.open(image_path)
            if img.format.lower() not in image_exts:
                os.remove(image_path)
        except:
            os.remove(image_path)
```

3. **Prepare data augmentation and train-validation split:**
```python
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_data = datagen.flow_from_directory("D:\\Depress", target_size=(48, 48), batch_size=64, class_mode='binary', subset='training')
val_data = datagen.flow_from_directory("D:\\Depress", target_size=(48, 48), batch_size=64, class_mode='binary', subset='validation')
```

4. **Define early stopping callback:**
```python
class EarlyStoppingCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy') >= 0.99:
            self.model.stop_training = True
```

5. **Build and compile CNN model:**
```python
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(48, 48, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
              loss="binary_crossentropy", metrics=["accuracy"])
```

6. **Train the model:**
```python
history = model.fit(train_data, validation_data=val_data, epochs=100, callbacks=[EarlyStoppingCallback()])
```

## Conclusion
This project provides a machine learning-based approach to detecting depression in students through facial analysis. Future enhancements include integrating real-time video analysis and expanding the dataset for improved accuracy.


## Usage
1. The user fills out a mental health questionnaire.
2. The system processes responses using NLP techniques.
3. The trained machine learning model predicts the probability of depression.
4. Results are displayed with severity levels and suggested mental health resources.

## Future Improvements
- Enhancing dataset quality by collecting real-world data.
- Incorporating deep learning models for improved accuracy.
- Expanding the system with multilingual support.
- Integrating with chatbot-based mental health support.

## License
This project is licensed under the MIT License.

## Contributors
- **Kangkan Patowary** (Developer)

## Contact
For queries or contributions, contact:
- Email: kangkanpatowary18@gmail.com
- GitHub: [kangkan10102001](https://github.com/kangkan10102001)
- LinkedIn: [kangkan-patowary](https://www.linkedin.com/in/kangkan-p-393166239/)





