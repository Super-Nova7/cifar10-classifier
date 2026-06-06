from flask import Flask, request
import tensorflow as tf
import numpy as np

app = Flask(name)

model = tf.keras.models.load_model("cifar10_model.keras")

classes = [
"airplane",
"automobile",
"bird",
"cat",
"deer",
"dog",
"frog",
"horse",
"ship",
"truck"
]

@app.route("/")
def home():
    return "CIFAR-10 Model Running"

@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]

    img = tf.keras.utils.load_img(
    file,
    target_size=(32, 32)
)

img = tf.keras.utils.img_to_array(img)

img = img / 255.0

img = np.expand_dims(img, axis=0)

prediction = model.predict(img)

predicted_class = classes[np.argmax(prediction)]

return {
    "prediction": predicted_class
}

if name == "main":
app.run(debug=True)