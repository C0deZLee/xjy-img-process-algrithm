import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

class mnistModel:
    def __init__(self, filename):
        self.filename = filename
        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        self.x = tf.placeholder(tf.float32, [None, 784])
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

        self.x_image = tf.reshape(self.x, [-1,28,28,1])
        self.h_conv1 = tf.nn.relu(conv2d(self.x_image, self.W_conv1) + self.b_conv1)
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

    def train(self):
        with tf.Session() as sess:
            sess.run(self.init_op)
            for i in range(20000):
                batch_xs, batch_ys = self.mnist.train.next_batch(50)
                if i % 100 == 0:
                    train_accuracy = self.accuracy.eval(feed_dict={
                        self.x: batch_xs, self.y_: batch_ys, self.keep_prob: 1.0})
                    print("step %d, training accuracy %d" %(i, train_accuracy * 100) + "%")
                self.train_step.run(feed_dict={self.x: batch_xs, self.y_: batch_ys, self.keep_prob: 0.5})
                
            save_path = self.saver.save(sess, "model.ckpt")
            print ("Model saved in file: ", save_path)

    def predict(self, img):
        img = 255 - img
        resized = cv2.resize(img, (28, 28), interpolation = cv2.INTER_CUBIC)[:, :, 0]
        resized = resized.reshape(1, -1)
        if (not(os.path.exists(self.filename + ".index"))):
            print("not exist")
            self.train()
        with tf.Session() as sess:
            sess.run(self.init_op)
            self.saver.restore(sess, "model.ckpt")
            # print ("Model restored.")  
            prediction = tf.argmax(self.y_conv, 1)[0]
            return prediction.eval(feed_dict = {self.x: resized, self.keep_prob: 1.0}, session=sess)

        
        



    