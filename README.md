# MNIST Handwritten Digit Classifier

This project is a **Deep Learning model** built to classify handwritten digits (0–9) using the **MNIST dataset**. The model is implemented in **Python** using **TensorFlow/Keras**, with fully connected (dense) layers, batch normalization, dropout, and LeakyReLU activations.

The purpose of this project is to demonstrate end-to-end deep learning workflow including **data loading, preprocessing, model building, training, visualization, and evaluation**.

---

## Colab Link

You can also run this project in Google Colab (interactive notebook):

[MNIST Colab Notebook](https://colab.research.google.com/drive/13o4lIqxAgYIcW5pmA_amlIdK8NG2O8KX?usp=sharing)

---

## Project Overview

### 1. Dataset Loading
- MNIST dataset is loaded directly from Keras (`tensorflow.keras.datasets.mnist`).
- Training set: 60,000 images (28x28 pixels, grayscale).
- Test set: 10,000 images.
- Initial exploration prints:
  - Shape of training and test images
  - Shape of labels
  - First image and its label
  - Distribution of classes in training set
- Observation: Slight class imbalance (~5400–6700 per digit), but dense networks handle this well, so no resampling is done.

### 2. Preprocessing
- Normalize pixel values to the range [0,1] by dividing by 255.0.
- Flatten 28x28 images into a 1D vector of 784 pixels for input to dense layers.
- Convert labels to **one-hot encoded vectors** (10 classes) for categorical cross-entropy loss.

### 3. Model Architecture
- **Input Layer:** 784 neurons (flattened 28x28 images)
- **Hidden Layer 1:** 512 neurons + BatchNormalization + LeakyReLU
- **Hidden Layer 2:** 256 neurons + BatchNormalization + LeakyReLU + Dropout(0.2)
- **Hidden Layer 3:** 128 neurons + BatchNormalization + LeakyReLU + Dropout(0.1)
- **Output Layer:** 10 neurons (softmax) for digit classification
- BatchNormalization stabilizes learning and speeds up convergence.
- LeakyReLU prevents dead neurons.
- Dropout reduces overfitting by randomly deactivating neurons during training.

### 4. Training
- Optimizer: **Adam** (adaptive learning rate)
- Loss: **categorical_crossentropy** (multi-class classification)
- Metrics: **accuracy**
- Training Parameters:
  - Epochs: 12
  - Batch Size: 128
  - Validation Split: 15% of training data
- During training, accuracy and loss are tracked for both training and validation sets.

### 5. Visualization
- Plots of **training vs validation accuracy** over epochs.
- Plots of **training vs validation loss** over epochs.
- Optional: Display 9 random test images with **predicted vs true labels**.

### 6. Evaluation
- Evaluate the model on the test set (`x_test_flat`) to report **test accuracy** and **test loss**.
- Make predictions on test images and visualize some random samples.
- The final test accuracy is usually **~97–98%** with this dense network architecture.

---

## How to Run Locally

### 1. Clone the Repository
```
git clone https://github.com/<username>/mnist-digit-classifier.git
cd mnist-digit-classifier
2. Create and Activate a Virtual Environment
Windows ():
python -m venv venv
venv\Scripts\Activate.ps1
Windows ():
python -m venv venv
venv\Scripts\activate.bat
Mac/Linux:
python -m venv venv
source venv/bin/activate
3. Install Dependencies
pip install -r requirements.txt
4. Run the Project
python main.py
The script will:
Load and preprocess MNIST data
Build and train the model
Plot training/validation accuracy and loss
Evaluate on test data
Display a few test images with predicted labels

Requirements
Minimum Python 3.8+, with dependencies listed in requirements.txt:
numpy>=1.23.0
matplotlib>=3.7.0
tensorflow>=2.15.0
Notes
The model uses a fully connected dense network, which is simple and easy to understand.
For higher accuracy, you could switch to a CNN (Convolutional Neural Network).
Slight class imbalance exists in MNIST but does not require resampling.
Training plots help monitor overfitting and learning progress.

Author
Built and documented by Toukir Ahamed Pigeon

