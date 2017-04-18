from loader import load_mnist_data, load_mnist_labels
import csv
import logging
import numpy as np
import math
import sys

LEARNING_RATE = 0.05
EPOCHS = 3000
REPORT = 150
TEST_BATCH_SIZE = 100
DROPOUT_P = 0.5

# Helper functions for output squashing and backprop
def tanh(in_array):
    """Computes the tanh function for a numpy array"""
    return np.tanh(in_array)

def tanh_(in_array):
    """Computes the derivative of the tanh function for a numpy array"""
    return 1-np.power(tanh(in_array),2)

class NN_Layer:

    def __init__(self, in_size, out_size):
        # Initialize trainable parameters
        self.weights = np.matrix(np.random.randn(in_size + 1, out_size) * 0.05)
        self.input = None # for convenience, store input to this layer
        self.output = None # for convenience, store output from this layer

    def feed_forward(self, input_vector):
        self.input = np.concatenate((input_vector,np.ones((input_vector.shape[0],1))), axis=1)
        self.output = np.dot(self.input, self.weights)
        return tanh(self.output)

class NN_Network:

    def __init__(self, sizes):
        # Initialize NN_Layer objects according to supplied sizes
        self.sizes = sizes
        self.layers = [NN_Layer(sizes[i], sizes[i+1]) for i in np.arange(len(sizes)-1)]

    def feed_forward(self, arr):
        """Computes the forward pass of a neural network given an input, and returns an np array"""
        for layer in self.layers:
            arr = layer.feed_forward(arr)
        return arr

    def backprop(self,output,expected):
        """Performs backpropagation through the neural network for one entry, using squared error"""
        error = expected-output # gradient of the quadratic cost function
        for layer in self.layers[::-1]:
            delta = np.multiply(error, tanh_(layer.output))
            error = np.dot(delta, layer.weights.T)[:,:-1]
            layer.weights += np.dot(layer.input.T, delta) * LEARNING_RATE

    def train(self, arr, epochs):
        """Takes a 2D array of floats as input, formatted with last item as labels and all others as attributes"""
        for x in range(epochs):
            forward = self.feed_forward(np.matrix(arr[:,:-1]))
            self.backprop(forward,arr[:,-1:])
            if x%REPORT == 0:
                logging.warning("fin l'epoch "+str(x))

    def test(self, arr):
        ff_result = self.feed_forward(arr[:,:-1])
        for inx in xrange(len(ff_result)):
            logging.error(str(ff_result[inx])+" : "+str(arr[inx,-1]))
        sum_error = np.sum(abs(ff_result-arr[:,-1:]))
        return sum_error/len(arr)

if __name__ == "__main__":
        # Do 20% testing and 80% training by default
        # get test data
        train_input = load_mnist_data('trainingimages', offset=0, batch_max=TEST_BATCH_SIZE)['data']
        train_label = load_mnist_labels('traininglabels', offset=0, batch_max=TEST_BATCH_SIZE)['data']
        data = np.concatenate((train_input, train_label), axis=1)
        if int(sys.argv[1]) != len(data[0])-1:
            raise ValueError("\033[91mInput layer ("+sys.argv[1]+") and data ("+str(len(data[0])-1)+") do not match. Terminating...\033[0m")
        training = data[:4*len(data)/5]
        test = data[4*len(data)/5:]
        neural_net = NN_Network([int(arg) for arg in sys.argv[1:]])
        neural_net.train(training, EPOCHS)
        print "\nAverage error on test is: \033[91m" + str(neural_net.test(test))
