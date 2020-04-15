import numpy as np
import cv2
import matplotlib.pyplot as plt
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import pickle
import os

class mnistModel:
    def __init__(self, filename):
        self.X_train, self.y_train = fetch_openml('mnist_784', version=1, return_X_y=True)
        self.X_train.reshape((len(self.X_train), -1))
        self.filename = filename
        self.model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=50000, alpha=1e-4,
                        solver='sgd', verbose=True, random_state=1, n_iter_no_change = 50000,
                        learning_rate_init=.1)
        # self.model = LogisticRegression(C=50. / self.X_train.shape[0], penalty='l1', solver='saga', tol=0.1)
        self.hasTrain = False

    def train(self):
        self.model.fit(self.X_train, self.y_train)
        print("The accuracy of training is: ")
        print(self.model.score(self.X_train, self.y_train))
        with open(self.filename, 'wb') as f:
            pickle.dump(self.model, f)
        self.hasTrain = True

    def predict(self, img):
        if (not(os.path.exists(self.filename))):
            self.train()
        if (not(self.hasTrain)):
            with open(self.filename, 'rb') as f:
                self.model = pickle.load(f)
        img = 255 - img
        resized = cv2.resize(img, (28, 28), interpolation = cv2.INTER_CUBIC)[:, :, 0]
        resized = resized.reshape(1, -1)
        return int(self.model.predict(resized)[0])

        
        



    