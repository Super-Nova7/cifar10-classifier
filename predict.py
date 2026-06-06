import tensorflow as tf
(xtrain,ytrain), (xtest, ytest) = tf.keras.datasets.cifar10.load_data()


model = tf.keras.models.load_model("cifar10_model.keras")
ss = model.predict(xtest[:3])
preds = tf.argmax(ss, axis=1)

for i in range(3):
    print(
        "Prediction:",
        class_names[preds[i]],
        "| Actual:",
        class_names[ytest[i][0]]
    )