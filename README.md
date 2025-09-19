# MNIST Digit Classifier

This project is a **deep learning model** that classifies handwritten digits (0–9) using the MNIST dataset. The model is built using **TensorFlow/Keras** with dense layers, batch normalization, and dropout regularization.

## Colab Link
You can also run this project in Google Colab: [MNIST Colab Notebook](https://colab.research.google.com/drive/13o4lIqxAgYIcW5pmA_amlIdK8NG2O8KX?usp=sharing)

## How to Run Locally

1. Clone the repository:
```bash
git clone https://github.com/<username>/mnist-digit-classifier.git
cd mnist-digit-classifier
2. Create and activate a virtual environment:
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
3. Install dependencies:
pip install -r requirements.txt
4. Run the project:
python main.py
5. The script will train the model, print test accuracy/loss, and save visualizations of predictions.
Notes
    Ensure you have Python 3.8+ installed.
    Adjust batch size and epochs in main.py according to your system capability.
    For GPU support, install tensorflow-gpu.