import tensorflow as tf
(xtrain, ytrain), (xtest, ytest) = tf.keras.datasets.cifar10.load_data()

xtrain = xtrain /255
xtest = xtest/255
inputs = tf.keras.layers.Input(shape=(32,32,3))
x = tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3), activation="relu",padding="same")(inputs)



x = tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3), activation="relu",padding="same")(x)
x = tf.keras.layers.MaxPooling2D((2,2))(x)

x=tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3), activation="relu",padding="same")(x)
x=tf.keras.layers.Conv2D(filters=128,kernel_size=(3,3),padding="same")(x)

x = tf.keras.layers.BatchNormalization() (x)

x = tf.keras.layers.Activation("relu") (x)

x = tf.keras.layers.MaxPooling2D((2,2))(x)

x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(128,activation="relu")(x)
x = tf.keras.layers.Dense(256,activation="relu") (x)
x = tf.keras.layers.Dropout(0.5) (x)
outputs = tf.keras.layers.Dense(10,activation="softmax")(x)
model = tf.keras.Model(inputs=inputs,outputs=outputs)
model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
model.fit(xtrain, ytrain, epochs=5,validation_split=0.2)
prediction = model.predict(xtest[:1])
print(prediction.argmax())
model.evaluate(xtest, ytest)
model.save("cifar10_model.keras")