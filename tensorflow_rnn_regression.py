# using tensorflow and RNN as a regressor.  Tensorflow backprop is a little different then typical backprop, google it to find out

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


BATCH_START = 0
TIME_STEPS = 20
BATCH_SIZE = 50
INPUT_SIZE = 1
OUTPUT_SIZE = 1
CELL_SIZE = 10
LR = 0.006


def get_batch():  # batch data is generated by this function
    global BATCH_START, TIME_STEPS
    # xs shape (50batch, 20steps)
    xs = np.arange(BATCH_START, BATCH_START+TIME_STEPS*BATCH_SIZE).reshape((BATCH_SIZE, TIME_STEPS)) / (10*np.pi)
    seq = np.sin(xs)
    res = np.cos(xs)
    BATCH_START += TIME_STEPS
    # plt.plot(xs[0, :], res[0, :], 'r', xs[0, :], seq[0, :], 'b--')
    # plt.show()
    # returned seq, res and xs: shape (batch, step, input)
    return [seq[:, :, np.newaxis], res[:, :, np.newaxis], xs]


class LSTMRNN(object):  # create the LSTM Cell, on the bottom and top cell we'll create a hidden layer for inputs and outputs.
    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size):
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        with tf.name_scope('inputs'):
            self.xs = tf.placeholder(tf.float32, [None, n_steps, input_size], name='xs')
            self.ys = tf.placeholder(tf.float32, [None, n_steps, output_size], name='ys')
        with tf.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.variable_scope('LSTM_cell'):
            self.add_cell()
        with tf.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(LR).minimize(self.cost)

    def add_input_layer(self):
        l_in_x = tf.reshape(self.xs, [-1, self.input_size], name = '2_2D')  # (batch * n_step, in_size), l_in_x means input layer in x, we have to reshape the placeholder from above into a 2D for x, the placeholder for x has 3D (None, n_steps, input_size).  It must be reshaped in order to use it in the Wx + b.  This is the calculation for the input layer
        Ws_in = self._weight_variable([self.input_size, self.cell_size])
        # bs (cell_size, )
        bs_in = self._bias_variable([self.cell_size])
        # l_in_y = (batch * n_steps, cell_size)
        with tf.name_scope('Wx_plus_b'):
            l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in  # now we have 2D time 2D which is alright, plus the biases.  All the variables are pick from what was defined above.
        # reshape l_in_y back into 3d in order to input back into the cell l_in_y -> (batch, n_steps, cell_size)
        self.l_in_y = tf.reshape(l_in_y, [-1, self.n_steps, self.cell_size], name = '2_3D')  # l_in_y is the output of layer 1, so we reshape the output a 3D tensor only then can we fit it into this cell.

    def add_cell(self):  # adding to the cell, we return the outputs and final states for the cell
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.cell_size, forget_bias = 1.0, state_is_tuple = True)
        with tf.name_scope('initial_state'):
            self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)  # the initial state is the zero state
        # we're going to add a loop about this cell
        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(lstm_cell, self.l_in_y, initial_state = self.cell_init_state, time_major = False)  # the loop is dynamic RNN, you use the LSTM cell and the outputs of the above layer, initial state is same as what we called above, time_major is false because time steps is the 2nd and not 1st or major dimension.  As a big picture idea, we take the self.cell_final_state and put that in the next batch as the initial state, self.cell_init_state and then run through the RNN

    def add_output_layer(self):  # we use the output to then become the input to the output layer
        # shape = (batch * steps, cell_size)
        l_out_x = tf.reshape(self.cell_outputs, [-1, self.cell_size], name = '2_2D')  # the inputs of the output layer is every output for the cell for every time step, since this a sequence (regression) and not a prediction of one image (classification) so we're going to use all outputs at once as the l_out_x or layer out x and we need to reshape this 2D because we are going to multiple by the weights which is 2D
        Ws_out = self._weight_variable([self.cell_size, self.output_size])
        bs_out = self._bias_variable([self.output_size])
        # shape = (batch * step, output_size)
        with tf.name_scope('Wx_plus_b'):
            self.pred = tf.matmul(l_out_x, Ws_out) + bs_out  # this is the prediction of every time step, it's not just one time step and every batch


    def compute_cost(self):  # we are going to use the tf.nn.seq2seq to compute the loss for every time step.  This is between the prediction and the target y's.  This is represented as a list of loss for every time step.  We calculate the loss for every step and every batch for the self.pred and self.ys
        losses = tf.nn.seq2seq.sequence_loss_by_example(
            [tf.reshape(self.pred, [-1], name='reshape_pred')],
            [tf.reshape(self.ys, [-1], name='reshape_target')],
            [tf.ones([self.batch_size * self.n_steps], dtype=tf.float32)],
            average_across_timesteps=True,
            softmax_loss_function=self.ms_error,  # use the mean square error because by default the loss function is calculated by softmax because this one is for the classification
            name='losses'
        )
        with tf.name_scope('average_cost'):  # going to calculate the average cost in this batch and then pass this cost into the training step on line 49: self.train_op = tf.train.AdamOptimizer(LR).minimize(self.cost) to minimize the cost
            self.cost = tf.div(
                tf.reduce_sum(losses, name='losses_sum'),  # we're going to sum up all the losses and ...
                tf.cast(self.batch_size, tf.float32),  # ... divide by the batch size to get the average cost of this batch
                name='average_cost')
            tf.scalar_summary('cost', self.cost)

    def ms_error(self, y_pre, y_target):  # now use the mean square error of the prediction and target
        return tf.square(tf.sub(y_pre, y_target))  # calculation method for the regressor

    def _weight_variable(self, shape, name='weights'):
        initializer = tf.random_normal_initializer(mean=0., stddev=1.,)
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def _bias_variable(self, shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)


if __name__ == '__main__':
    model = LSTMRNN(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE)
    sess = tf.Session()
    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter("logs", sess.graph)
    sess.run(tf.global_variables_initializer())
    # relocate to the local dir and run this line to view it on Chrome (http://0.0.0.0:6006/):
    # $ tensorboard --logdir='logs'

###  If you comment out all the following below you can evaluate the model in tensorboard   ###
## In these following steps,
    plt.ion()  # to show and not block the main screen
    plt.show()
    for i in range(200):
        seq, res, xs = get_batch()  # this xs is the time point for all sequence and all results
        if i == 0:  # for the initial time steps, we just pass for the feed dictionary, model x is sequence and model y is result, for the first step i==0 we don't create any states but for all that follows we do.
            feed_dict = {
                    model.xs: seq,
                    model.ys: res,
                    # create initial state below
            }
        else:
            feed_dict = {
                model.xs: seq,
                model.ys: res,
                model.cell_init_state: state    # use last state as the initial state for this run but it is the final state of the calculation and puts the final state as the model.cell_init_state
            }

        _, cost, state, pred = sess.run(
            [model.train_op, model.cost, model.cell_final_state, model.pred],  # model.cell_final_state taken from above final state and it becomes the initial state for the next round
            feed_dict=feed_dict)

        # plotting, we create one animation to show how the RNN learns to fit in the targets
        plt.plot(xs[0, :], res[0].flatten(), 'r', xs[0, :], pred.flatten()[:TIME_STEPS], 'b--')  # for every loop in every step we're going to plot the time steps.  results uses a red color and xs[0, :], pred.flatten()[:TIME_STEPS], 'b--' is the prediction of the first example/sequence in the batch. xs[0, :], res[0].flatten() is the real data
        plt.ylim((-1.2, 1.2))
        plt.draw()
        plt.pause(0.3)  # we pause 0.3 seconds for every plot

        if i % 20 == 0:  # for every 20 steps we print out the cost
            print('cost: ', round(cost, 4))
            result = sess.run(merged, feed_dict)
            writer.add_summary(result, i)
