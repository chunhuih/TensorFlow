# This is the code to perform toxicity prognostics using machine learning
# It uses the Google TensorFlow framework to perform machine learning

import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# this is the number of independents
num_independents = 30
# x is the vector containing all the independent variables
x = tf.placeholder(tf.float32, [None, num_independents])
# W is the matrix that will be filled by the algorithm
W=tf.Variable(tf.zeros([num_independents, 5]))
# b is a vector that will also be filled by the algorithm
b=tf.Variable(tf.zeros([5]))
# y is the toxicity level. y can be 0, 1, 2, 3, and 4.
# y = W * x + b
# y is the vector for the predicted values from calculations using the formula above.
y=tf.nn.softmax(tf.matmul(x,W) + b)
print(x)
# y_ is the true distribution from actual data.
y_ = tf.placeholder(tf.float32,[None, 5])
# the cross entropy is to be minimized.
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# tf.reduce_mean() takes the average of the sums
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
init=tf.initialize_all_variables()
sess=tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
