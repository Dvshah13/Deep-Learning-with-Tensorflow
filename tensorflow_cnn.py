import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data # import the data set

# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def compute_accuracy(v_xs, v_ys):  # this is the accuracy of the network, which is what we output
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)  # the truncated normal is similar to the random normal distribution, once we pass the shape in we will get the weights.
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)  # the biases could be different, we define the initial biases set equal to 0.1, when training this 0.1 will change to another value.  Normally the bias is a small positive value.  Once we pass the shape in we will get the biases.
    return tf.Variable(initial)

def conv2d(x, W):  # start to define the CNN layer.  x is the inputs into the layer and W is the weights.  Tensorflow makes it pretty easy to define the convolutional layer.
    #  stride [1, x_movement, y_movement, 1] or [1,1,1,1]
    #  must have strides[0] = strides[4] = 1
    return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'SAME')  # this is a 2 dimensional CNN.  We pass in the x inputs which is the whole image and then pass the weights (W) in.  We also have to pass the strides in where you detach a patch/kernal from the image.  In tensorflow, you represent this as an array, where the first and last value has to be 1.  This [1,1,1,1] and has 4 locations.  The second value will have a stride of 1 in the x direction/horizontal direction, the third value is for the y direction/vertical which is also 1.  First and last value must = 1.  There are two methods for padding, one is valid, one is the same, here we use the same.  The valid padding is always in the image and the same padding has part of the patch outside of this image and we will fill the outer area as 0.  For the same padding, the output has the same shape (height and width) of the original image.  The valid padding will have a smaller height and width for output.

def max_pool_2x2(x):  #  in order to avoid a big loss of information in a big stride, we will use pooling.  We decrease the stride but use pooling to achieve the same output shape.  Note the bigger the stride the smaller the output size and shape but we lose information this way, thus the pooling helps us resolve this and get the smaller output size and shape we desire without the information loss.  Tensorflow makes it pretty easy to define pooling.
    #  stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME' )  # you can use max or average pooling, we use max here. the x here is the return of conv2d, the k-size is used to sample down different length vectors into the same length before the fully connecte layer, the strides still has the 1st and 4th set to 1 but in max pooling we move 2 pixels for every step, that is how we compress, we use the pooling to reduce the size and not the CNN step.  The same padding is used as before.  The steps here are very similar to the convolutional net function.

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784]) # 28x28
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
# pre-processing our xs data
x_image = tf.reshape(xs, [-1,28,28,1])  # define the array, because the xs data includes the samples number, so the -1 is to ingnore the dimension for samples number first, then add this dimension automatically. 28, 28 is for all the pixels in this image.  1 is for the image channel because all of our images are black and white, the channel is 1 if our images were color the channel can be 3 (rgb)
# print x_image.shape # if we want to print the shape of this x_image, the result of this should be [n_smaples, 28, 28, 1]. In our example, -1 represents the n_smaples.

## conv1 layer ##
W_conv1 = weight_variable([5,5,1,32])  # W represents the weight, weight variable is from our function we defined earlier.  [5,5,1,32], The patch is 5x5 (length x width), input size (this is the thickness of the image) is 1 and output size should be 32 (which is the length at the end, remember we are trying to compress something to be taller and smaller in length and width).
b_conv1 = bias_variable([32])  # b represents the bias, bias variable is from our function we defined earlier.  The shape should be 32, the same as the output size of the image.
# now we can compute the first convolutional neural network, conv2d function represents that, this is a hidden layer we name it h_conv1
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # very similar to the Wx + b, take in x_image, W_conv1 = weights, b_conv1 - biases.  Before we output from this layer, we can do a non-linearlization that is the tf.nn.relu that we add.  We use RELU to this layer.  Output size = 28 x 28 x 32 because we have a same padding, the image shape is the same as the output shape in here which is 28 x 28, however the height changes to 32, thickness increases.
# now we can do the pooling.
h_pool1 = max_pool_2x2(h_conv1)  # the input for pooling is h_conv1, the x in max_pool_2x2 is the return of the processed conv2d which we need to pass the value to pooling after the activation function.  h_pool1 is the overall output of this conv1.  The size after pooling can be changed, the x and y movement in conv1 = 1 but in pooling the movement = 2, when we have double the step size or movement, the size of the output image shrinks to 1/2 the size.  output size 14 x 14 x 32, length and width are half but the height doesn't change

## conv2 layer ## Copy from conv1 layer but change to 2.
W_conv2 = weight_variable([5,5,32,64])  # patch 5 x 5, input size 32, output size 64. Compared to conv1, we keep the patch size the same but the input size becomes 32 which is the output size of the last layer, we assume the output for this layer has a height of 64, you just want to keep increasing the thickness or height of those layers.  The original image has a thickness/height of 1, after the first cnn layer we have a height of 32, after the second we have a height of 64.
b_conv2 = bias_variable([64])  # same as the output size of the image after cnn layer 2
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # output size 14 x 14 x 64 after running the second layer.
h_pool2 = max_pool_2x2(h_conv2)  # output size = 7 x 7 x 64

# Here is also where we pass things to the fully connected layer
## func1 layer ##  This is to add some functional layers
W_fc1 = weight_variable([7*7*64, 1024])  # W_function1, we pass in the shape of the weights which is 7 x 7 x 64 and we assume the output size to be 1024 to make it wider.
b_fc1 = bias_variable([1024])  # b_function1, we pass in the output size
# we have to reshape the value from h_pool2 and flatten the pool
h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64])  # h_pool2 = [n_samples, 7, 7, 64] -> [n_samples, 7*7*64], we changed the shape of h_pool2 to 7*7*64 to multiply the reuslts to the weights
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  # hidden layer added, taking in the all the functions
# to consider overfitting, add a dropout function
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## func2 layer ##  Very similar to the first func1 layer so I just copy it and modify
W_fc2 = weight_variable([1024, 10])  # now the input size is 1024 and the output size has 10 units for 10 digits
b_fc2 = bias_variable([10])  # the output from the W_fc1 passed in
# note we don't need the pooling, hidden layer or droupout here since this is our last step but we have to now have add a prediction
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)  # we pass in the drop function and take in the W_fc2 and b_fc2



# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))       # loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)  # used the AdamOptimizer as opposed to the gradient descent optimizer because it will work better in this case.  When using AdamOptimizer we have to pass an even smaller learning rate into adam so here we use 1e-4.

sess = tf.Session()
# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 50 == 0:
        print(compute_accuracy(
            mnist.test.images, mnist.test.labels))
