from loader import load_mnist_data, load_mnist_labels
import csv
import logging
import numpy as np
import math
import sys

LEARNING_RATE = 0.05
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
        self.output = np.dot(self.input, self.weight)
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
            error = np.dot(delta, layer.weights.T)
            layer.weights += np.dot(layer.input.T, delta) * LEARNING_RATE

    def train(self, arr):
        """Takes a 2D array of floats as input, formatted with last item as labels and all others as attributes"""
        for _ in range(50):
            for row in arr:
                forward = self.feed_forward(np.matrix(row[:-1]))
                self.backprop(forward,row[-1])

    def test(self, arr):
        sum_error = 0
        for row in arr:
            ff_result = float(self.feed_forward(row[:-1]))
            print str(row[-1]) + "  -->  " + str(ff_result)
            sum_error += abs(row[-1]-ff_result)
        return sum_error/len(arr)

if __name__ == "__main__":
    if "-fold" != sys.argv[1]:
        # Do 20% testing and 80% training by default
        data = load_csv("data/verify.csv")
        if int(sys.argv[1]) != len(data[0])-1:
            raise ValueError("\033[91mInput layer ("+sys.argv[1]+") and data ("+str(len(data[0])-1)+") do not match. Terminating...\033[0m")
        training = data[:4*len(data)/5]
        test = data[4*len(data)/5:]
        neural_net = NN_Network([int(arg) for arg in sys.argv[1:]])
        neural_net.train(training)
        print "\nAverage error on test is: \033[91m" + str(neural_net.test(test))
    else:
        try:
            fold_number = int(sys.argv[2])
        except ValueError:
            raise SyntaxError("Syntax error in specifying number of folds.")
        data = load_csv("data/verify.csv")
        if int(sys.argv[3]) != len(data[0])-1:
            raise ValueError("\033[91mInput layer ("+sys.argv[3]+") and data ("+str(len(data[0])-1)+") do not match. Terminating...\033[0m")
        for fold in range(fold_number):
            neural_net = NN_Network([int(arg) for arg in sys.argv[3:]])
            training1 = data[:fold*len(data)/fold_number]
            test = data[fold*len(data)/fold_number : (fold+1)*len(data)/fold_number]
            training2 = data[(fold+1)*len(data)/fold_number:]
            neural_net.train(training1)
            neural_net.train(training2)
            print "\nAverage error on fold #" + str(fold) + " test is: " + str(neural_net.test(test))