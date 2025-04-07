#!/usr/bin/env python3
"""Main file to trigger the project."""
###############################################################################
# Project - Convolutional Neural Network Model
# Filename - CNN.py
# Arguments -
# Created By - Vivek Salgia
# Creation Date - 04/03/2025
# Reviewed By -
# Reviewed Date -
# Change logs -
# Version   Date         Type   Changed By                  Comments
# =======   ============ ====   ==============  ===============================
# 1.0       04/03/2025   I     Vivek Salgia    Initial Creation
###############################################################################

import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import time


def unpickle(file):
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


def plot_sample(X, y, index, classes):
    plt.figure(figsize=(15, 2))
    plt.imshow(X[index])
    plt.xlabel(classes[int(y[index])])
    key_pressed = plt.waitforbuttonpress()
    # print(y[index])
    print(classes[int(y[index])])


def get_train_set():
    encoded_raw_data = {}
    labels = []
    data = [[]]
    for i in range(5):
        filename = "cifar-10-batches-py/data_batch_" + str(i + 1)
        # print(filename)
        encoded_raw_data = unpickle(filename)
        labels = np.append(labels, encoded_raw_data[b"labels"])
        # dataeach = encoded_raw_data[b"data"]
        # print(f"Dataeach shape - {dataeach.shape}")
        if i == 0:
            data = encoded_raw_data[b"data"]
        else:
            data = np.concatenate((data, encoded_raw_data[b"data"]), axis=0)

    data2D = np.reshape(data, (50000, 3, 1024))
    datatrnx = np.transpose(data2D, (0, 2, 1))
    data3D = np.reshape(datatrnx, (50000, 32, 32, 3))
    return labels, data3D


def train_model(X_train, y_train):
    # Create Model
    cnn = models.Sequential()
    cnn.add(layers.Input(shape=(32, 32, 3)))
    # Convolution layer
    cnn.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu"))
    cnn.add(layers.MaxPooling2D((2, 2)))

    cnn.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
    cnn.add(layers.MaxPooling2D((2, 2)))

    # Dense
    cnn.add(layers.Flatten())
    cnn.add(layers.Dense(64, activation="relu"))
    cnn.add(layers.Dense(10, activation="softmax"))

    cnn.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    cnn.fit(X_train, y_train, epochs=10)
    cnn.save("cnn_model.keras")


def predict(X_test):
    # load model
    cnn = models.load_model("cnn_model.keras")
    y_pred = cnn.predict(X_test)
    return y_pred


if __name__ == "__main__":
    # print(os.listdir("cifar-10-batches-py"))
    # Read Data
    labels, data = get_train_set()
    labels = labels.astype(np.uint8)

    # Normalize the values
    data3DNormalized = data / 255

    # print(f"{data2D[0]}")
    # print(f"{datatrnx[0].shape}")
    # print(f"data3D shape - {data3D.shape}")

    # train_model(data3DNormalized, labels)

    # Read Labels
    encoded_raw_labels = unpickle("cifar-10-batches-py/batches.meta")
    names = np.array(encoded_raw_labels[b"label_names"])

    # List Comprehension
    decoded_names = [index.decode() for index in names]
    # print(decoded_names)
    # plot_sample(data, labels, 0, decoded_names)

    # Read Test Data Set
    encoded_raw_data = unpickle("cifar-10-batches-py/test_batch")
    testData = encoded_raw_data[b"data"]
    testLabels = encoded_raw_data[b"labels"]

    print(f"testData Shape - {testData.shape}")

    tData2D = np.reshape(testData, (10000, 3, 1024))
    tDatatrnx = np.transpose(tData2D, (0, 2, 1))
    tData3D = np.reshape(tDatatrnx, (10000, 32, 32, 3))

    y_pred = predict(tData3D)
    plot_sample(tData3D, testLabels, 2, decoded_names)
    # print(f"Actual Image - {decoded_names[testLabels[0]]}")
    print(f"Predicted Image - {decoded_names[np.argmax(y_pred[2])]}")
