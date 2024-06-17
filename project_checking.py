import tensorflow as tf
import numpy as np
import cv2
import json

#file_path = r'C:\MyProject\fake_2.jpg'
file_path = r'C:\MyProject\tendulkar image.webp'
labels = ['real', 'fake']
with open('fakevsreal_model.json', 'r') as json_file:
    json_savedModel = json_file.read()

# load the model weights
model = tf.keras.models.model_from_json(json_savedModel)
model.load_weights('fakevsreal_weights.h5')
model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=["accuracy"])


def read_and_preprocess(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # reading the image
    img = cv2.resize(img, (128, 128))  # resizing it to 128*128
    img = np.array(img, dtype='float32')  # convert its datatype so that it could be normalized
    img = img / 255  # normalization (now every pixel is in the range of 0 and 1)
    return img


img = read_and_preprocess(file_path)
img = np.expand_dims(img, 0)
predictions = model.predict(img)  # predicting the label
predicted_class = np.argmax(predictions, axis=1)

# Print the predicted class
if predicted_class == 1:
    print("Predicted: Fake")
else:
    print("Predicted: Real")

