# Image Classification with TensorFlow and OpenCV

![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Overview

This project demonstrates how to build and train an image classification model using **TensorFlow** and **OpenCV**. It showcases the key steps for data preparation, model architecture design, training, and evaluation, enabling users to classify images effectively.

## Features

- **Image Preprocessing**: Utilizes OpenCV for image resizing, normalization, and augmentation.
- **Convolutional Neural Network (CNN)**: Constructs a CNN architecture using TensorFlow/Keras.
- **Model Training**: Implements training loops with validation and early stopping.
- **Visualization**: Includes plots for training progress such as accuracy and loss curves.
- **Modular Code**: Organized into functions and classes for better readability and reusability.

## Installation

### Prerequisites

- Python 3.12
- pip (Python package installer)

### Install Dependencies

Run the following command to install the required packages:

```bash
pip install tensorflow opencv-python matplotlib
```

## Dataset
The dataset for this project can be sourced from **Kaggle** or other repositories. It should be organized into folders where each folder represents a class, containing images of that class. Place the dataset in the `data/` directory before running the notebook.

Example folder structure:
```
data/
    class1/
        image1.jpg
        image2.jpg
    class2/
        image3.jpg
        image4.jpg
```


## Usage

1. **Clone the Repository**

   ```bash
   git clone https://github.com/dhruvsorathiya25/image-classification-tensorflow.git
   cd image-classification-tensorflow
   ```

2. **Prepare the Dataset**

   - Place your dataset in the `data/` directory.
   - Ensure the dataset is organized into subdirectories for each class.

3. **Open the Jupyter Notebook**
   ```bash
   jupyter notebook image_cla.ipynb
   ```
4. **Follow the Notebook Steps**
   - **Data Preprocessing**: Load and preprocess images using OpenCV.
   - **Model Building**: Define the CNN architecture.
   - **Training**: Train the model with the prepared dataset.
   - **Evaluation**: Assess the model's performance using validation data.

## Project Structure

- `image_cla.ipynb`: Main Jupyter Notebook containing code and explanations.
- `README.md`: Project documentation (you're reading it).
- `data/`: Folder to store your dataset.
- `models/`: Folder to save trained models.

## Building the CNN Model

### Model Architecture

The CNN model is built using TensorFlow's Keras API. Below is an overview of the architecture:

1. **Input Layer**

   - Accepts images of shape `(height, width, channels)`.

2. **Convolutional Layers**

   - **Conv2D**: Applies convolution filters to extract features.
   - **Activation (ReLU)**: Introduces non-linearity.
   - **MaxPooling2D**: Reduces spatial dimensions.

3. **Dropout Layers**

   - Prevents overfitting by randomly dropping neurons during training.

4. **Flatten Layer**

   - Converts 2D feature maps into a 1D feature vector.

5. **Fully Connected (Dense) Layers**
   - **Dense**: Learns complex patterns.
   - **Activation (ReLU)**: Non-linear activation.
   - **Output Layer**: Uses `softmax` activation for multi-class classification.

### Example Code Snippet

```python
import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn_model(input_shape, num_classes):
    model = models.Sequential()

    # Convolutional Block 1
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))

    # Convolutional Block 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Convolutional Block 3
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten and Dense Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))

    # Output Layer
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

# Example usage
input_shape = (128, 128, 3)  # Example input shape
num_classes = 10             # Example number of classes
model = build_cnn_model(input_shape, num_classes)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
```

### Training the Model

```python
history = model.fit(
    train_dataset,
    epochs=25,
    validation_data=validation_dataset,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    ]
)
```

### Evaluation

After training, evaluate the model's performance on the test dataset and visualize metrics such as accuracy and loss.

```python
import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
