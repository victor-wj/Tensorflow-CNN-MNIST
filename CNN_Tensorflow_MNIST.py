import os
import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data

current_directory   = os.path.dirname(os.path.realpath(__file__))

# download the MNIST data in <current directory>/MNIST_data/
#one_hot=True: one-hot-encoding, means only return the highet probability
mnist = input_data.read_data_sets(current_directory + "/MNIST_data/", one_hot=True)

# make it as default session so we do not need to pass sess
sess = tf.InteractiveSession()

# X is placeholder for 28 x 28 image data
X = tf.placeholder(tf.float32, shape=[None, 784])

# y_ is a 10 element ventor, it is the predicted probability of each digit class, e.g. [0, 0, 0.12, 0, 0, 0, 0.98, 0, 0.1, 0]
y_ = tf.placeholder(tf.float32, [None, 10])

# change the MNIST input data from a list to a 28 x 28 x 1 grayscale value cube, for CNN using
x_image = tf.reshape(X, [-1, 28, 28, 1], name="x_image")

def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# convolution
def conv2D(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# max pooling to control overfitting
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 1st convolution layer
W_conv1 = weight_variable([5, 5, 1, 32]) # one filter is 5 x 5, 32 features for each filter
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2D(x_image, W_conv1) + b_conv1) # first convolution, use RELU activation
h_pool1 = max_pool_2x2(h_conv1) # first max pooling, output is 14 x 14 image ( [28, 28] / 2 = [14, 14])

# 2nd convolution layer
W_conv2 = weight_variable([5, 5, 32, 64]) # one filter is 5 x 5, 64 features for each filter
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2D(h_pool1, W_conv2) + b_conv2) 
h_pool2 = max_pool_2x2(h_conv2) # second max pooling, output is 7 x 7 image ( [14, 14] / 2 = [7, 7])

# fully connected layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# the final layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

# define the model
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# loss measurement
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))

# loss optimization
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# accuracy
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# initialize all variables
sess.run(tf.global_variables_initializer())

# train the model
num_steps = 2500
display_every = 10
batch_size = 50
dropout = 0.5

start_time = time.time()
end_time = time.time()

for i in range(num_steps):
    batch = mnist.train.next_batch(batch_size)
    train_step.run(feed_dict={X: batch[0], y_:batch[1], keep_prob:dropout})
    
    if i%display_every == 0:
        train_accuracy = accuracy.eval(feed_dict={X:batch[0], y_:batch[1], keep_prob:1.0})
        end_time = time.time()
        print ("step {0}, elapsed time {1:.2f} seconds, training accuracy {2:.3f}%".format(i, end_time-start_time, train_accuracy*100.0))

# test model
print("test accuracy {0:.3}%".format(accuracy.eval(feed_dict={X:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0}) * 100.0))