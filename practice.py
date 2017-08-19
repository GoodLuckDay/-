import tensorflow as tf
import numpy as np
xy = np.loadtxt('data-04-zoo.csv',delimiter=',',dtype=np.float32)
x_data = xy[:,0:-1]
y_data = xy[:,[-1]]

nb_count = 7
X = tf.placeholder(tf.float32,shape=[None, 16])
Y = tf.placeholder(tf.int32, shape=[None, 1])

Y_hot_one = tf.one_hot(Y, nb_count)
Y_hot_one = tf.reshape(Y_hot_one,[-1,nb_count])
W = tf.Variable(tf.random_normal([16,nb_count]),name='weight')
b = tf.Variable(tf.random_normal([nb_count]), name='bias')

logic = tf.matmul(X,W) + b
hypothesis = tf.nn.softmax(logits=logic)

cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logic, labels=Y_hot_one)
cost = tf.reduce_mean(cost_i)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

predicted = tf.arg_max(hypothesis,1)
correct_prediction = tf.equal(predicted, tf.arg_max(Y_hot_one,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run([hypothesis,predicted], feed_dict={X:x_data, Y:y_data}))


