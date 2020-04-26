import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# import tensorflow_datasets as tfds

# from tensorflow.examples.tutorials.mnist import input_data

def shuffle_in_unison(a, b):
    assert a.shape[0] == b.shape[0]
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(a.shape[0])
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

def normalize(img):
    minv = np.min(img)
    maxv = np.max(img)
    img = img * (img - minv) / (maxv - minv)
    return np.where(img > 0.3, 1, 0)

class mnistModel:
    def __init__(self, filename, datadir, warm_start):
        tf.reset_default_graph()
        self.filename = filename
        self.warm_start = warm_start
        self.X_train = np.zeros((1, 28, 28, 1))
        self.y_train = np.zeros((1, 10))
        self.X_test = 0
        self.y_test = 0
        self.datadir = datadir
        self.x = tf.placeholder(tf.float32, [None, 28, 28, 1])
        self.y_ = tf.placeholder(tf.float32, [None, 10])
        # self.W = tf.Variable(tf.zeros([784, 10]))
        # self.b = tf.Variable(tf.zeros([10]))
        # self.y = tf.nn.softmax(tf.matmul(self.x, self.W) + self.b)

        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)

        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)


        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1], padding='SAME')

        self.W_conv1 = weight_variable([5, 5, 1, 32])
        self.b_conv1 = bias_variable([32])

        self.h_conv1 = tf.nn.relu(conv2d(self.x, self.W_conv1) + self.b_conv1)
        self.h_pool1 = max_pool_2x2(self.h_conv1)


        self.W_conv2 = weight_variable([5, 5, 32, 64])
        self.b_conv2 = bias_variable([64])

        self.h_conv2 = tf.nn.relu(conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)
        self.h_pool2 = max_pool_2x2(self.h_conv2)

        self.W_fc1 = weight_variable([7 * 7 * 64, 1024])
        self.b_fc1 = bias_variable([1024])

        self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, 7*7*64])
        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1)

        self.keep_prob = tf.placeholder(tf.float32)
        self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)

        self.W_fc2 = weight_variable([1024, 10])
        self.b_fc2 = bias_variable([10])

        self.y_conv=tf.nn.softmax(tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2)

        self.cross_entropy = -tf.reduce_sum(self.y_ * tf.log(self.y_conv))
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)
        self.correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        self.init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

    def getdata(self):
        for i in range(10):
            filedir = os.path.join(self.datadir, str(i))
            print(filedir)
            for files in os.walk(filedir):
                X_train = np.zeros((min(10000, len(files[2])), 28, 28, 1))
                y_train = np.zeros((min(10000, len(files[2])), 10))
                for j in range(min(10000, len(files[2]))):
                    if (j % 1000 == 0):
                        print("finish loading: " + str(j) + "/" + str(min(10000, len(files[2]))))
                    img = cv2.imread(os.path.join(filedir, files[2][j]))
                    if img is None:
                        print(files[2][j])
                        continue
                    img = cv2.resize(normalize(img), (28, 28), interpolation = cv2.INTER_AREA)
                    X_train[j] = img[:, :, 0:1]
                    y_train[j][i] = 1 
                if (i == 0):
                    self.X_train = X_train
                    self.y_train = y_train
                else:                  
                    self.X_train = np.concatenate((self.X_train, X_train))
                    self.y_train = np.concatenate((self.y_train, y_train))
            print(self.X_train.shape)
        self.X_train, self.y_train = shuffle_in_unison(self.X_train, self.y_train)
        self.X_test = self.X_train[95000:]
        self.y_test = self.y_train[95000:]
        self.X_train = self.X_train[:95000]
        self.y_train = self.y_train[:95000]


    def train(self):
        sess = tf.Session()
        with tf.Session() as sess:
            sess.run(self.init_op)
            if (self.warm_start and os.path.exists(self.filename + ".meta")):
                print("Model restored from " + self.filename)
                self.saver.restore(sess, self.filename)
            idx = 0
            best_loss = 10000
            for i in range(7500):
                batch_xs, batch_ys = self.X_train[idx:min(idx+500, self.X_train.shape[0])], self.y_train[idx:min(idx+500, self.X_train.shape[0])]
                idx += 500
                if (idx >= self.X_train.shape[0]):
                    idx = 0
                if i % 100 == 0:
                    train_accuracy = self.accuracy.eval(feed_dict={
                        self.x: batch_xs, self.y_: batch_ys, self.keep_prob: 1.0})
                    test_accuracy = self.accuracy.eval(feed_dict={
                        self.x: self.X_test, self.y_: self.y_test, self.keep_prob: 1.0})
                    print("step %d, training accuracy %3f, test accuracy %3f" %(i, train_accuracy, test_accuracy))
                _, loss = sess.run((self.train_step, self.cross_entropy), feed_dict={self.x: batch_xs, self.y_: batch_ys, self.keep_prob: 0.5})
                if i % 100 == 0:
                    print("The loss function is: %f" %(loss))
                    if (loss < best_loss):
                        save_path = self.saver.save(sess, self.filename)
                        print ("Model saved in file:", save_path)
                        best_loss = loss         

    def predict(self, img):
        #print(resized.shape)
        resized = cv2.resize(img, (28, 28), interpolation = cv2.INTER_AREA)
        #print(resized.shape)
        resized = resized[:, :, 0:1]
        resized = np.expand_dims(resized, 0)
        if (not(os.path.exists(self.filename + ".meta"))):
            print("not exist")
            self.train()
        sess = tf.Session()
        with tf.Session() as sess:
            sess.run(self.init_op)
            self.saver.restore(sess, self.filename)
            # print ("Model restored.")  
            prediction = tf.argmax(self.y_conv, 1)[0]
            prob = tf.reduce_max(self.y_conv, 1)[0]
            self.saver.restore(sess, self.filename)
            prob_, prediction_ = sess.run((prob, prediction), feed_dict = {self.x: resized, self.keep_prob: 1.0})
            return (prob_, prediction_)


      
        



    
