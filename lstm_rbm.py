import tensorflow as tf
import numpy as np
import glob

from tensorflow.python.ops import control_flow_ops
from tqdm import tqdm

import RBM
import midi_manipulation


"""
    This file contains the TF implementation of the RNN-RBM, as well as the hyperparameters of the model
"""

note_range         = midi_manipulation.span #The range of notes that we can produce
n_visible          = 2*note_range*midi_manipulation.num_timesteps #The size of each data vector and the size of the RBM visible layer
n_hidden           = 50 #The size of the RBM hidden layer
n_hidden_recurrent = 100 #The size of each RNN hidden layer

class LSTMNet():
    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, n_visible]) # Holds song data
        self.lr = tf.placeholder(tf.float32) # learning rate
        self.size_bt = tf.shape(self.x)[0] # batch size

        self.W = tf.Variable(tf.zeros([n_visible, n_hidden]), name="W")
        self.Wuh = tf.Variable(tf.zeros([n_hidden_recurrent, n_hidden]), name="Wuh")
        self.Wuv = tf.Variable(tf.zeros([n_hidden_recurrent, n_visible]), name="Wuv")

        self.Wu_f = tf.Variable(tf.zeros([n_visible + n_hidden_recurrent, n_hidden_recurrent]), name="Wu_f")
        self.Wu_i = tf.Variable(tf.zeros([n_visible + n_hidden_recurrent, n_hidden_recurrent]), name="Wu_i")
        self.Wu_o = tf.Variable(tf.zeros([n_visible + n_hidden_recurrent, n_hidden_recurrent]), name="Wu_o")
        self.WC = tf.Variable(tf.zeros([n_visible + n_hidden_recurrent, n_hidden_recurrent]), name="WC")

        # TODO delete
        self.Wvu = tf.Variable(tf.zeros([n_visible, n_hidden_recurrent]), name="Wvu")
        self.Wuu = tf.Variable(tf.zeros([n_hidden_recurrent, n_hidden_recurrent]), name="Wuu")

        self.bh = tf.Variable(tf.zeros([1, n_hidden]), name="bh")
        self.bv = tf.Variable(tf.zeros([1, n_visible]), name="bv")

        self.bf = tf.Variable(tf.zeros([1, n_hidden_recurrent], name="bf"))
        self.bi = tf.Variable(tf.zeros([1, n_hidden_recurrent], name="bi"))
        self.bo = tf.Variable(tf.zeros([1, n_hidden_recurrent], name="bo"))
        self.bC = tf.Variable(tf.zeros([1, n_hidden_recurrent], name="bC"))

        # TODO delete
        self.bu  = tf.Variable(tf.zeros([1, n_hidden_recurrent]), name="bu")

        self.curr_lstm_state = tf.Variable(tf.zeros([1, n_hidden_recurrent]), name="curr_state")

        self.u0  = tf.Variable(tf.zeros([1, n_hidden_recurrent]), name="u0")
        self.BH_t = tf.Variable(tf.zeros([1, n_hidden]), name="BH_t")
        self.BV_t = tf.Variable(tf.zeros([1, n_visible]), name="BV_t")

        self.cost = self.inference()

    def training_vars(self):
        return self.W, self.Wuh, self.Wuv, self.Wu_f, self.Wu_i, self.Wu_o, self.WC, self.bh, self.bv, self.bf, self.bi, self.bo, self.bC, self.u0

    # a different version of lstm_recurrence that just returns new_state as vector
    def lstm_recurrence_adjusted(self, utm1, sl, old_state):
        sl = tf.reshape(sl, [1, n_visible])
        inputs = tf.concat(1, [utm1, sl])

        ft = tf.sigmoid(self.bf + tf.matmul(inputs, self.Wu_f))
        it = tf.sigmoid(self.bi + tf.matmul(inputs, self.Wu_i))

        Cstar = tf.tanh(self.bC + tf.matmul(inputs, self.WC))

        new_state = tf.mul(ft, old_state) + tf.mul(it, Cstar)

        ot = tf.sigmoid(self.bo + tf.matmul(inputs, self.Wu_o))
        ut = tf.mul(ot, tf.tanh(new_state))

        return ut, new_state

    def lstm_recurrence(self, utm1, sl):
        sl = tf.reshape(sl, [1, n_visible])

        inputs = tf.concat(1, [utm1, sl])

        ft = tf.sigmoid(self.bf + tf.matmul(inputs, self.Wu_f))
        it = tf.sigmoid(self.bi + tf.matmul(inputs, self.Wu_i))

        Cstar = tf.tanh(self.bC + tf.matmul(inputs, self.WC))
        new_state = tf.mul(ft, self.curr_lstm_state) + tf.mul(it, Cstar)

        ot = tf.sigmoid(self.bo + tf.matmul(inputs, self.Wu_o))
        ut = tf.mul(ot, tf.tanh(new_state))

        self.curr_lstm_state = new_state
        return ut

    def rnn_recurrence(self, u_tm1, sl):
        #Iterate through the data in the batch and generate the values of the RNN hidden nodes
        sl  =  tf.reshape(sl, [1, n_visible])
        u_t = (tf.tanh(self.bu + tf.matmul(sl, self.Wvu) + tf.matmul(u_tm1, self.Wuu)))
        return u_t

    def visible_bias_recurrence(self, bv_t, u_tm1):
        #Iterate through the values of the RNN hidden nodes and generate the values of the visible bias vectors
        bv_t = tf.add(self.bv, tf.matmul(u_tm1, self.Wuv))
        return bv_t

    def hidden_bias_recurrence(self, bh_t, u_tm1):
        #Iterate through the values of the RNN hidden nodes and generate the values of the hidden bias vectors
        bh_t = tf.add(self.bh, tf.matmul(u_tm1, self.Wuh))
        return bh_t       

    def generate_recurrence(self, count, k, u_tm1, primer, x, music, state):
        #This function builds and runs the gibbs steps for each RBM in the chain to generate music
        #Get the bias vectors from the current state of the RNN
        bv_t = tf.add(self.bv, tf.matmul(u_tm1, self.Wuv))
        bh_t = tf.add(self.bh, tf.matmul(u_tm1, self.Wuh))

        #Run the Gibbs step to get the music output. Prime the RBM with the previous musical output.
        x_out = RBM.gibbs_sample(primer, self.W, bv_t, bh_t, k=25)
        
        #Update the RNN hidden state based on the musical output and current hidden state.
        u_t, new_state = self.lstm_recurrence_adjusted(u_tm1, x_out, state)
        
        # u_t  = (tf.tanh(self.bu + tf.matmul(x_out, self.Wvu) + tf.matmul(u_tm1, self.Wuu)))

        #Add the new output to the musical piece
        music = tf.concat(0, [music, x_out])
        return count+1, k, u_t, x_out, x, music, new_state

    def generate(self, num, prime_length=100):
        """
            This function handles generating music. This function is one of the outputs of the build_rnnrbm function
            Args:
                num (int): The number of timesteps to generate
                x (tf.placeholder): The data vector. We can use feed_dict to set this to the music primer. 
                size_bt (tf.float32): The batch size
                u0 (tf.Variable): The initial state of the RNN
                n_visible (int): The size of the data vectors
                prime_length (int): The number of timesteps into the primer song that we use befoe beginning to generate music
            Returns:
                The generated music, as a tf.Tensor

        """
        Uarr = tf.scan(self.rnn_recurrence, self.x, initializer=self.u0)
        U = Uarr[np.floor(prime_length/midi_manipulation.num_timesteps).astype(np.int32), :, :]
        count = tf.constant(1, tf.int32)
        k = tf.constant(num)
        primer = tf.zeros([1, n_visible], tf.float32)
        music = tf.zeros([1, n_visible], tf.float32)

        state = tf.zeros([1, n_hidden_recurrent], tf.float32)

        [_, _, _, _, _, music, state] = tf.while_loop(lambda count, num_iter, *args: count < num_iter,
                                                         self.generate_recurrence, [count, k, U, primer, self.x, music, state],
                                                                                                                 # TODO: put correct dims in shape_invariants
                                                                                                                 shape_invariants=[count.get_shape(), k.get_shape(), U.get_shape(), primer.get_shape(), self.x.get_shape(), tf.TensorShape([None, n_visible]), state.get_shape()], parallel_iterations=1)
        return music

    def inference(self):
        #Reshape our bias matrices to be the same size as the batch.
        tf.assign(self.BH_t, tf.tile(self.BH_t, [self.size_bt, 1]))
        tf.assign(self.BV_t, tf.tile(self.BV_t, [self.size_bt, 1]))

        #Scan through the rnn and generate the value for each hidden node in the batch
        u_t  = tf.scan(self.lstm_recurrence, self.x, initializer=self.u0)

        #Scan through the rnn and generate the visible and hidden biases for each RBM in the batch
        self.BV_t = tf.reshape(tf.scan(self.visible_bias_recurrence, u_t, tf.zeros([1, n_visible], tf.float32)), [self.size_bt, n_visible])
        self.BH_t = tf.reshape(tf.scan(self.hidden_bias_recurrence, u_t, tf.zeros([1, n_hidden], tf.float32)), [self.size_bt, n_hidden])

        #Get the free energy cost from each of the RBMs in the batch 
        return RBM.get_free_energy_cost(self.x, self.W, self.BV_t, self.BH_t, k=15)
