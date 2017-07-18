import time
import tensorflow as tf
import numpy as np
import pandas as pd
from scipy.misc import imread
from alexnet import AlexNet
import pickle
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

n_classes = 43
EPOCHS = 5
BATCH_SIZE = 128
# TODO: Load traffic signs data.
# Load pickled data

with open("train.p", mode='rb') as f:
    data = pickle.load(f)

# TODO: Fill this in based on where you saved the training and testing data    
X_train, X_valid, y_train, y_valid = train_test_split(data['features'], data['labels'], test_size=0.2, random_state=0)
# TODO: Split data into training and validation sets.

# TODO: Define placeholders and resize operation.
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
resized = tf.image.resize_images(x, (227, 227))
one_hot_y = tf.one_hot(y, n_classes)
# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)

# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)
# TODO: Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], n_classes)
fc8W = tf.Variable(tf.truncated_normal(shape, stddev=1e-2))
fc8b = tf.Variable(tf.zeros(n_classes))
logits = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
loss_op = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = 0.01)
training_operation = optimizer.minimize(loss_op)

# TODO: Train and evaluate the feature extraction model.
correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    session = tf.get_default_session()
    total_accuracy = 0
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x = X_data[offset: offset + BATCH_SIZE]
        batch_y = y_data[offset: offset + BATCH_SIZE]
        accuracy = session.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

 ## Train the model
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print ("Training the model")
    print ()
    
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        num_examples = len(X_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end_offset = offset + BATCH_SIZE
            batch_X = X_train[offset: end_offset]
            batch_Y = y_train[offset: end_offset]
            sess.run(training_operation, feed_dict={x:batch_X, y: batch_Y})
        training_accuracy = evaluate(X_train, y_train)
        validation_accuracy = evaluate(X_valid, y_valid)
        
        print ("Epoch {} ....".format(i))
        print ("Training accuracy : {:.2f}".format(training_accuracy))
        print ("Validation accuracy : {:.2f}".format(validation_accuracy))
        print()

