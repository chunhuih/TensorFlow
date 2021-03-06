import tensorflow as tf
import numpy as np

#create 100 phony x, y data points in NumPy y = x * 0.1 + 0.3
#x_data = np.random.rand(100).astype(np.float32) * 1
x_data = np.array(list(range(100))).astype(np.float32)*.01
y_data = x_data * 0.1 + 0.3
print x_data
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

loss = tf.reduce_mean(tf.square(y - y_data))
# The parameter for this optimizer function is crucial for the success of optimization.
# Experiments show that the optimal value depends on the magnitude of x values.
optimizer = tf.train.GradientDescentOptimizer(.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in range(601):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b))

