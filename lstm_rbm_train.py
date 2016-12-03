import time
import sys
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import lstm_rbm
import midi_manipulation 

"""
    This file contains the code for training the RNN-RBM by using the data in the Pop_Music_Midi directory
"""


batch_size = 100 #The number of trianing examples to feed into the rnn_rbm at a time
epochs_to_save = 5 #The number of epochs to run between saving each checkpoint
saved_weights_path = "parameter_checkpoints/initialized.ckpt" #The path to the initialized weights checkpoint file

def main(num_epochs):
    # First, build model then get pointers to params
    neural_net = lstm_rbm.LSTMNet()
    tvars = neural_net.training_vars()
    cost = neural_net.cost
    x = neural_net.x
    lr = neural_net.lr

    # Construct optimizers & gradient computation
    opt_func = tf.train.GradientDescentOptimizer(learning_rate=lr)
    gvs = opt_func.compute_gradients(cost, tvars)
    gvs = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in gvs] #We use gradient clipping to prevent gradients from blowing up during training
    updt = opt_func.apply_gradients(gvs) #The update step involves applying the clipped gradients to the model parameters

    songs = midi_manipulation.get_songs('Pop_Music_Midi') # Load corpus of songs

    saver = tf.train.Saver(tvars)
    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init) 
        saver.restore(sess, saved_weights_path) #Here we load the initial weights of the model that we created with weight_initializations.py

        #We run through all of the songs n_epoch times
        print "starting"
        for epoch in range(num_epochs):
            costs = []
            start = time.time()
            for s_ind, song in enumerate(songs):
                for i in range(1, len(song), batch_size):
                    tr_x = song[i:i + batch_size] 
                    alpha = min(0.01, 0.1/float(i)) #We decrease the learning rate according to a schedule.
                    _, C = sess.run([updt, cost], feed_dict={neural_net.x: tr_x, neural_net.lr: alpha}) 
                    costs.append(C) 
            #Print the progress at epoch
            print "epoch: {} cost: {} time: {}".format(epoch, np.mean(costs), time.time()-start)
            print
            #Here we save the weights of the model every few epochs
            if (epoch + 1) % epochs_to_save == 0: 
                saver.save(sess, "parameter_checkpoints/epoch_{}.ckpt".format(epoch))

if __name__ == "__main__":
    main(int(sys.argv[1]))


