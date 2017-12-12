#NW를 이용을 한 MNIST
import tensorflow as tf
import matplotlib.pyplot as plt
import random
tf.set_random_seed(777)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

np_classes = 10
X = tf.placeholder(tf.float32, shape=[None, 784])
Y = tf.placeholder(tf.float32, shape=[None, np_classes])
W1 = tf.Variable(tf.random_normal([784, 70]),name='weight1')
b1 = tf.Variable(tf.random_normal([70]), name='bias1')
layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([70, 70]),name='weight2')
b2 = tf.Variable(tf.random_normal([70]), name='bias2')
layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)

W3 = tf.Variable(tf.random_normal([70, 70]),name='weight3')
b3 = tf.Variable(tf.random_normal([70]), name='bias3')
layer3 = tf.sigmoid(tf.matmul(layer2, W3) + b3)

W4 = tf.Variable(tf.random_normal([70, np_classes]),name='weight4')
b4 = tf.Variable(tf.random_normal([np_classes]), name='bias4')
hypothesis = tf.sigmoid(tf.matmul(layer3, W4) + b4)
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(cost)

is_correct = tf.equal(tf.arg_max(hypothesis,1), tf.arg_max(Y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, dtype=tf.float32))

traning_epochs = 100
batch_size = 100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(traning_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c,_ = sess.run([cost, optimizer], feed_dict={X:batch_xs, Y:batch_ys})
            avg_cost = avg_cost + c / total_batch
        print('Epoch : ', '%04d' % (epoch + 1), 'cost = ', '{:.9f}'.format(avg_cost))
    print("Accuracy : ", sess.run(accuracy, feed_dict={X:mnist.test.images, Y:mnist.test.labels}))
    r = random.randint(0, mnist.test.num_examples - 1)
    print('Label: ', sess.run(tf.arg_max(mnist.test.labels[r:r+1], 1)))
    print('Prediction: ', sess.run(tf.arg_max(hypothesis,1) ,feed_dict={X:mnist.test.images[r:r+1]}))
    plt.imshow(mnist.test.images[r: r+1].reshape(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()