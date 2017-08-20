import tensorflow as tf
tf.set_random_seed(777)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

np_classes = 10
X = tf.placeholder(tf.float32, shape=[None, 784])
Y = tf.placeholder(tf.float32, shape=[None, np_classes])
W = tf.Variable(tf.random_normal([784,np_classes]), name='weight')
b = tf.Variable(tf.random_normal([np_classes]), name='bias')
hypothesis = tf.matmul(X,W) + b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(cost)

traning_epoches = 15
batch_size = 100

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(traning_epoches):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X:batch_xs, Y:batch_ys}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch

    print('Epoch: ', '%04d' % (epoch + 1), 'cost= ', '{:.9f}'.format(avg_cost))
print('Learning Finished!')

correct_prediction = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))
print('Accuracy: ', sess.run(accuracy, feed_dict={X:mnist.test.images, Y: mnist.test.labels}))