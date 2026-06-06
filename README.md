CIFAR-10 Image Classification API

Project Overview

This project uses a Convolutional Neural Network (CNN) trained on the CIFAR-10 dataset to classify images into one of the following categories:

- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

The model is served using Flask and can receive an image through an API endpoint and return the predicted class.

---

Technologies Used

- Python
- TensorFlow / Keras
- Flask
- NumPy

---

Project Structure

project/

├── main.py

├── cifar10_model.keras

├── requirements.txt

└── README.md

---

Installation

Clone the repository:

git clone <repository_url>
cd project

Install dependencies:

pip install -r requirements.txt

---

Run the Application

python main.py

The server will start at:

http://127.0.0.1:5000

---

API Endpoint

Health Check

GET /

Response:

CIFAR-10 Model Running

---

Predict Image

POST /predict

Send an image file with key:

image

Example Response:

{
  "prediction": "dog"
}

---

Model Information

Dataset:

- CIFAR-10

Architecture:

- Convolutional Neural Network (CNN)

Output Classes:

- 10 Classes

Saved Model Format:

- .keras

