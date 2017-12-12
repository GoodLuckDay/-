import tensorflow as tf
import numpy as np

def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-7)


data = np.array([[6,185],[8,218],[5,175],[10,336],[9,417],[2,61],[9,528],[10,818],[5,439],[2,40],[6,147],[7,448],[10,348],[7,393],[7,82],[9,443],[4,59]])
x_train = data[:,0]
y_train = data[:,[-1]]

W = tf.Variable(tf.random_normal([1]),name = 'weight')
b = tf.Variable(tf.random_normal([1]),name = 'bias')

hypothesis = x_train * W + b
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))
print(sess.run(hypothesis))