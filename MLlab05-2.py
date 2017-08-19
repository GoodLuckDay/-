import tensorflow as tf
file_queue = tf.train.string_input_producer(['data-03-diabetes.csv'],shuffle=False,name='file_queue')
read = tf.TextLineReader()
key, value = read.read(file_queue)

xy = tf.decode_csv(value, record_defaults=[[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.]])

train_x_batch, train_y_batch = tf.train.batch([xy[0:-1], xy[-1:]],batch_size=10)

X = tf.placeholder(tf.float32,shape=[None, 8])
Y = tf.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.random_normal([8,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

hypothesis = tf.sigmoid(tf.matmul(X, W) + b)
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

protected = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(hypothesis, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    for step in range(10001):
        x_data, y_data = sess.run([train_x_batch, train_y_batch])
        sess.run(train, feed_dict={X:x_data, Y:y_data})
        if step % 200 == 0:
            print(step, sess.run(cost,feed_dict={X:x_data, Y:y_data}))
    h, c, a = sess.run([hypothesis, protected,accuracy], feed_dict={X:x_data,Y:y_data})
    print("\nHypothesis : ", h, "\nCorrect (Y) : ", c, "\nAccuracy : ", a)
    coord.request_stop()
    coord.join()


