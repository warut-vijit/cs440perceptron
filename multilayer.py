from loader import load_mnist_data, load_mnist_labels
import csv
import logging
import numpy as np
import sys
import random
from matplotlib import pyplot

LEARNING_RATE = 0.001
EPOCHS = 30000
REPORT = 100
TEST_BATCH_SIZE = 100
TRAIN_BATCH_SIZE = 100
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
        self.weights = np.array(np.random.randn(in_size + 1, out_size) * 0.05)
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

    def train(self, examples, labels, epochs):
        """Takes a 2D array of floats as input, formatted with last item as labels and all others as attributes"""
        for x in range(epochs):
            forward = self.feed_forward(examples)
            self.backprop(forward, labels)

if __name__ == "__main__":
    # Do 20% testing and 80% training by default
    # get test data and train
    results = []
    neural_net = NN_Network([int(arg) for arg in sys.argv[1:]])
    for k in xrange(EPOCHS / REPORT):
        offset = random.randint(0, 1000-TRAIN_BATCH_SIZE)
        train_input = load_mnist_data('trainingimages', offset=offset, batch_max=TRAIN_BATCH_SIZE)['data']
        train_label = load_mnist_labels('traininglabels', offset=offset, batch_max=TRAIN_BATCH_SIZE)['data'] # get label
        train_label = np.array([[x==y[0] for x in xrange(10)] for y in train_label])
        neural_net.train(train_input, train_label, REPORT)
        logging.error("Finished epoch %s" % (k * REPORT))
        # test
        test_input = load_mnist_data('testimages', offset=0, batch_max=TEST_BATCH_SIZE)['data']
        test_label = load_mnist_labels('testlabels', offset=0, batch_max=TEST_BATCH_SIZE)['data']
        test_sum = 0
        labels = neural_net.feed_forward(test_input)
        for index in xrange(len(test_label)):
            test_sum += (max(enumerate(labels[index]), key=lambda b:b[1])[0] == test_label[index])
        results.append(float(test_sum)/TEST_BATCH_SIZE)
    pyplot.plot(results)
    pyplot.show()
