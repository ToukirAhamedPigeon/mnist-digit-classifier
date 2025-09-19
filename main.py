# Loading required Libraries
import numpy as np                                                                 # NumPy for numerical operations, arrays, reshaping, math functions
import matplotlib.pyplot as plt                                                    # Matplotlib for plotting graphs (accuracy, loss, sample digits visualization)
import random                                                                      # For Random Testing in the final testing with test data
import tensorflow as tf                                                            # TensorFlow as the main deep learning framework
from tensorflow import keras                                                       # Keras high-level API inside TensorFlow (easy model building and training)
from tensorflow.keras import layers                                                # Layers provide building blocks (Dense, Dropout, BatchNormalization, etc.)
from tensorflow.keras.utils import to_categorical                                  # to_categorical to convert labels (0–9) into one-hot encoded format
from tensorflow.keras.datasets import mnist                                        # Load the MNIST dataset directly from Keras
from tensorflow.keras.models import Sequential                                     # Sequential API for building models layer by layer
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU, Dropout  # Specific layer types for fully connected layers, batch normalization, and dropout

# Custom function to print data in a nice way in result
def custom_print(label,values): 
  print(label)
  print("")
  print(values)
  print("")
  print("")
  print("")

## 1. Dataset Loading
# Load the MNIST dataset and split into training and testing sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()   # x = images, y = labels

# Print the shape (dimensions) of training images
custom_print('Shape of x_train (number of training samples, image height, image width):', x_train.shape)

# Print the shape (dimensions) of training labels
custom_print('Shape of y_train (number of labels corresponding to training images):', y_train.shape)

# Print the shape (dimensions) of test images
custom_print('Shape of x_test (number of testing samples, image height, image width):', x_test.shape)

# Print the shape (dimensions) of test labels
custom_print('Shape of y_test (number of labels corresponding to testing images):', y_test.shape)

# Print the total number of training samples
custom_print('Total number of training samples in x_train:', len(x_train))

# Print the first image in training data (28x28 pixel values, grayscale between 0–255)
custom_print('First training image pixel matrix (28x28 grayscale values):', x_train[0])

# Print the actual label (digit) corresponding to the first training image
custom_print('True label (digit) of the first training image in y_train:', y_train[0])

# Count unique values and their counts in y_train
unique_values, counts = np.unique(y_train, return_counts=True)

# Print the unique values and their counts
print("The unique values and their counts of y_train:")
for value, count in zip(unique_values, counts):
    print(f"Value: {value}, Count: {count}")
# Print the total number of training samples
print(f"Total number of samples in y_train: {len(y_train)}")

# Observation: MNIST train data is slightly imbalanced
# The imbalance is very small (~5400–6700 per class)
# Deep learning models (Dense) can handle this tiny imbalance easily
# So I am leaving the dataset as-is, no resampling needed

print("Observation: MNIST training data is slightly imbalanced.")
print("The imbalance is very small (5400–6700 samples per class).")
print("Deep learning models like Dense networks can handle this tiny imbalance easily.")
print("Therefore, the dataset will be left as-is, no resampling needed.")

## 2. Preprocessing

# Normalize the training and test images by dividing pixel values by 255.0
# Original pixel values range from 0–255 (grayscale). After normalization, they range from 0–1.
# Normalization helps the neural network train faster and perform better.
x_train = x_train / 255.0
x_test = x_test / 255.0

# Flatten each 28x28 image into a 1D vector of length 784 (28*28 = 784).
# This is required because our dense neural network expects a flat input vector instead of a 2D image.
# -1 inside the reshape functions ensures, it will take the length of the data (60K in this case) dynamically.
x_train_flat = x_train.reshape((-1, 28*28))
x_test_flat = x_test.reshape((-1, 28*28))

# Define the number of classes (digits 0 through 9, total 10 classes).
num_classes = 10

# Convert the labels (y_train, y_test) into one-hot encoded format.
# Example: digit '3': [0,0,0,1,0,0,0,0,0,0]
# This makes the labels suitable for training with categorical_crossentropy loss.
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# Show the shape of flattened training data (number_of_samples × 784).
# This confirms that each image is now represented as a 1D vector of 784 values.
custom_print('Shape of flattened training images (samples × 784 pixels):', x_train_flat.shape)

# Show the shape of one-hot encoded labels (number_of_samples × 10).
# This confirms that each label is now represented as a one-hot vector of length 10.
custom_print('Shape of one-hot encoded training labels (samples × 10 classes):', y_train_cat.shape)

# Print the actual pixel values of the first training image after flattening.
# Since we used x_train_flat, the image is no longer a 28x28 matrix.
# It is now represented as a 1D vector of length 784 (all pixel values between 0 and 1).
custom_print('Flattened pixel values (1D vector of length 784) of the first training image:', x_train_flat[0])

# Print the one-hot encoded label for the first training image.
# Instead of showing a single digit (e.g., 5), it displays a vector of length 10.
# Example: digit '5': [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
custom_print('One-hot encoded label vector (length 10) for the first training image:', y_train_cat[0])


## 3. Model Architecture

tf.keras.backend.clear_session() # Clearing sessions before re run
# Build a Deep Neural Network with 3 hidden layers, BatchNormalization, Dropout, and LeakyReLU activations
model = Sequential([
    # Input layer: Flattened MNIST images (28x28 = 784 pixels)
    layers.Input(shape=(28*28,)),  
    # -------------------------- First Hidden Layer --------------------------
    Dense(512),                 # Fully connected layer with 512 neurons
    BatchNormalization(),       # Normalize outputs of previous layer to speed up training and stabilize learning
    LeakyReLU(alpha=0.01),      # LeakyReLU allows small gradient when neuron is not active (prevents dead neurons)

    # -------------------------- Second Hidden Layer -------------------------
    Dense(256),                 # Fully connected layer with 256 neurons
    BatchNormalization(),       # Normalize layer outputs
    LeakyReLU(alpha=0.01),      # Non-linear activation to learn complex patterns
    Dropout(0.2),               # Drop 20% of neurons randomly for regularization

    # -------------------------- Third Hidden Layer --------------------------
    Dense(128),                 # Fully connected layer with 128 neurons
    BatchNormalization(),       # Normalize layer outputs
    LeakyReLU(alpha=0.01),      # Activation function to improve gradient flow
    Dropout(0.1),               # Drop 10% of neurons

    # -------------------------- Output Layer -------------------------------
    Dense(num_classes, activation='softmax')  # Output layer with 10 neurons (digits 0–9)
                                              # Softmax converts raw scores to probabilities
])

# Print the model summary
model.summary()

# -------------------------- Custom Print --------------------------
print("\n--- Model Architecture Overview ---")
print("Input Layer: 28x28 pixels flattened to 784 features")
print("Hidden Layer 1: 512 neurons, BatchNorm, LeakyReLU")
print("Hidden Layer 2: 256 neurons, BatchNorm, LeakyReLU, Dropout(0.2)")
print("Hidden Layer 3: 128 neurons, BatchNorm, LeakyReLU, Dropout(0.1)")
print("Output Layer: 10 neurons (digit classes 0-9) with Softmax activation")
print("BatchNormalization helps stabilize learning and speeds up convergence")
print("LeakyReLU prevents dead neurons and allows small gradient flow")
print("Dropout prevents overfitting by randomly deactivating neurons during training\n")


### 4. Training
# ================================
# Model Compilation & Training
# ================================

# Compile the model: configure the learning process
model.compile(
    optimizer='adam',                        # Adam optimizer: adaptive learning rate, works well for most problems
    loss='categorical_crossentropy',         # Loss function: categorical_crossentropy for multi-class classification
    metrics=['accuracy']                     # Metrics to monitor: here we track accuracy during training
)

# ================================
# Train the model
# ================================

# Set training parameters
epochs = 12                                  # Number of times the model will see the entire training dataset
batch_size = 128                             # Number of samples per gradient update

# Train the model while keeping aside a part of the training data for validation
history = model.fit(
    x_train_flat,                            # Input training data (flattened 28x28 images)
    y_train_cat,                             # Output labels in one-hot encoded format
    validation_split=0.15,                   # Reserve 15% of training data for validation (used to check overfitting)
    epochs=epochs,                           # Number of training iterations over the entire dataset
    batch_size=batch_size,                   # Number of samples per batch to compute gradient and update weights
    verbose=2                                # Verbosity mode: 2 = one line per epoch showing progress
)


## 5. Visualization

# Plot training vs validation accuracy and loss
plt.figure(figsize=(12,5))  # Create a figure with width 12 inches and height 5 inches

# -------------------------------
# Accuracy Plot
# -------------------------------
plt.subplot(1,2,1)  # First subplot: 1 row, 2 columns, position 1
plt.plot(history.history['accuracy'], label='Train Accuracy')       # Plot training accuracy over epochs
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')  # Plot validation accuracy over epochs
plt.title('Model Accuracy Over Epochs')  # Add a descriptive title
plt.xlabel('Epoch')                       # Label for x-axis
plt.ylabel('Accuracy')                    # Label for y-axis
plt.legend()                              # Show legend to distinguish train vs val curves

# -------------------------------
# Loss Plot
# -------------------------------
plt.subplot(1,2,2)  # Second subplot: 1 row, 2 columns, position 2
plt.plot(history.history['loss'], label='Train Loss')          # Plot training loss over epochs
plt.plot(history.history['val_loss'], label='Validation Loss') # Plot validation loss over epochs
plt.title('Model Loss Over Epochs')  # Add a descriptive title
plt.xlabel('Epoch')                  # Label for x-axis
plt.ylabel('Loss')                   # Label for y-axis
plt.legend()                         # Show legend to distinguish train vs val curves

plt.tight_layout()  # Adjust subplots to prevent overlap
plt.show()          # Display the figure

# -------------------------------
# Custom descriptive prints
# -------------------------------
print("The left plot shows how the model's accuracy improves over epochs for both training and validation sets.")
print("The right plot shows how the model's loss decreases over epochs, indicating learning progress.")


## 6. Evaluation

# -----------------------------
# Evaluate the model on the test set
# -----------------------------
test_loss, test_acc = model.evaluate(x_test_flat, y_test_cat, verbose=0)     # Compute loss and accuracy on unseen test data
print(f'Test accuracy: {test_acc:.4f}')                                   # Print the test accuracy in 4 decimal places
print(f'Test loss: {test_loss:.4f}')                                      # Print the test loss in 4 decimal places

# -----------------------------
# Make predictions on test data
# -----------------------------
preds = model.predict(x_test_flat)                                           # Get predicted probabilities for each class
pred_labels = np.argmax(preds, axis=1)                                       # Convert probabilities to predicted class labels (0-9)

# -----------------------------
# Visualize a few test images with their true and predicted labels
# -----------------------------                                                            

print("\nDisplaying 9 random test images with model predictions:")

plt.figure(figsize=(10,6))                                                   # Set the figure size
for i, idx in enumerate(random.sample(range(len(x_test)), 9)):               # Select 9 random indices from the test set
    plt.subplot(3,3,i+1)                                                     # Create a 3x3 subplot
    plt.imshow(x_test[idx], cmap='gray')                                      # Show the image in grayscale
    plt.title(f'True: {y_test[idx]} | Pred: {pred_labels[idx]}')             # Display both true and predicted labels
    plt.axis('off')                                                           # Turn off axes for better visualization

plt.tight_layout()                                                            # Adjust spacing between subplots
plt.show()                                                                    # Render the plot
