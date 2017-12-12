#W값이 점차 개선이 되어가는 것을 볼수있다.

import  tensorflow as tf

X = [1, 2, 3]
Y = [1, 2, 3]

W = tf.Variable(tf.random_normal([1]), name='weight')

hypothesis = X * W

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(100):
    print(step, sess.run(W))
    sess.run(train)