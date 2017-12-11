#coding=utf-8

import time
import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

#load data
MNIST = input_data.read_data_sets("data/mnist", one_hot=True)

#define param
learning_rate = 0.01
batch_size = 128
n_epochs = 25

#create placeholder
X = tf.placeholder(tf.float32, [batch_size, 784], name="input_image")
Y = tf.placeholder(tf.float32, [batch_size, 10], name="out_number")

#create weights and bias 正态分布输出随机值
w = tf.Variable(tf.random_normal(shape=[784, 10], stddev=0.01), name="weights")
b = tf.Variable(tf.zeros([1, 10]), name="bias")

logits = tf.matmul(X, w) + b