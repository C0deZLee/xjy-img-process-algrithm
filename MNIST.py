import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
import pickle
import os

class mnistModel:
    def train(self, pkl_filename):
        digit = fetch_mldata("MNIST original", data_home = './datasets')
        X_train = digit["data"]
        y_train = digit["target"]
        X_train.reshape((len(X_train), -1))
        model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=30000, alpha=1e-4,
                    solver='sgd', verbose=10, random_state=1, n_iter_no_change = 30000,
                    learning_rate_init=.1)
        model.fit(X_train, y_train)
        with open(pkl_filename, 'wb') as f:
            pickle.dump(model, f)

    def predict(self, img, pkl_filename):
        if (not(os.path.exists(pkl_filename))):
            self.train(pkl_filename)
        with open(pkl_filename, 'rb') as f:
            pickle_model = pickle.load(f)
        img = 255 - img
        resized = cv2.resize(img, (28, 28), interpolation = cv2.INTER_CUBIC)[:, :, 0]
        resized = resized.reshape(1, -1)
        return int(pickle_model.predict(resized)[0])

        
        



    