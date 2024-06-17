import seaborn as sns
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

IMAGE_SIZE = 128
BATCH_SIZE = 32
CHANNELS = 3


def read_and_preprocess(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # reading the image
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))  # resizing it to 128*128
    img = np.array(img, dtype='float32')  # convert its datatype so that it could be normalized
    img = img / 255  # normalization (now every pixel is in the range of 0 and 1)
    return img


labels = ['real', 'fake']

X = []  # To store images
Y = []  # To store labels

folder_path = r'C:\MyProject\FakeRealdataset'

for image in os.scandir(folder_path):
    for entry in os.scandir(image.path):
        X.append(read_and_preprocess(entry.path))

        if image.name[0] == 'r':
            Y.append(0)  # real
        else:
            Y.append(1)  # fake

X = np.array(X)
X.shape  # We have 1289 image samples in total
print(X)

Y = np.array(Y)
Y.shape

real_count = len(Y[Y == 0])
fake_count = len(Y[Y == 1])

X_train, X_val, Y_train, Y_val = train_test_split(X, Y,
                                                  test_size=0.2,
                                                  shuffle=True,
                                                  stratify=Y,
                                                  random_state=123)

X_test, X_val, Y_test, Y_val = train_test_split(X_val, Y_val,
                                                test_size=0.5,
                                                shuffle=True,
                                                stratify=Y_val,
                                                random_state=123)

X_train.shape
X_test.shape

input_shape = (IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes = 2

model = tf.keras.Sequential([
    Conv2D(filters=32, kernel_size=(2, 2), activation='relu', input_shape=input_shape),
    MaxPooling2D((4, 4)),

    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPooling2D((3, 3)),
    Dropout(0.3),  # for regularization

    Conv2D(filters=64, kernel_size=(4, 4), activation='relu', padding='same'),
    Conv2D(filters=128, kernel_size=(5, 5), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Dropout(0.4),

    Conv2D(filters=128, kernel_size=(5, 5), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Dropout(0.5),

    Flatten(),  # flattening for feeding into ANN
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dense(n_classes, activation='softmax')
])

model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=["accuracy"])

earlystopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
checkpointer = ModelCheckpoint(filepath="fakevsreal_weights.h5", verbose=1, save_best_only=True)

history = model.fit(X_train, Y_train, epochs=10, validation_data=(X_val, Y_val), batch_size=BATCH_SIZE, shuffle=True,
                    callbacks=[earlystopping, checkpointer])

validation_loss, validation_accuracy = model.evaluate(X_val, Y_val)

print(f"Validation Accuracy: {validation_accuracy * 100:.2f}%")
print(f"Validation Loss: {validation_loss}")

test_loss, test_accuracy = model.evaluate(X_train, Y_train)

print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss}")

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

model_json = model.to_json()
with open("fakevsreal_model.json", "w") as json_file:
    json_file.write(model_json)

with open('fakevsreal_model.json', 'r') as json_file:
    json_savedModel = json_file.read()

# load the model weights
model = tf.keras.models.model_from_json(json_savedModel)
model.load_weights('fakevsreal_weights.h5')
model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=["accuracy"])

predictions = model.predict(X_test)

predict = []
for i in predictions:
    predict.append(np.argmax(i))

predict = np.asarray(predict)

accuracy = accuracy_score(Y_test, predict)
print(accuracy)

# plot the confusion matrix

cf_matrix = confusion_matrix(Y_test, predict)
plt.figure(figsize=(9, 7))

group_names = ['Real Images Predicted Correctly', 'Real Images Predicted as Fake', 'Fake Images Predicted as Real',
               'Fake Images Predicted Correctly']
group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
labels = np.asarray(labels).reshape(2, 2)
sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
plt.show()

report = classification_report(Y_test, predict)
print(report)
