from loader_adv import load_mnist_data, load_mnist_labels
import csv
import logging
import numpy as np
import sys
import random
from matplotlib import pyplot as plt

LEARNING_RATE = 0.002
EPOCHS = 1000
REPORT = 100
TEST_BATCH_SIZE = 100
TRAIN_BATCH_SIZE = 100
IMG_SIZE = 28

g_layers = [10, 784]
#d_layers = [784, 40, 30, 10]
d_layers = [784, 200, 10]

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
        self.input = np.concatenate((input_vector,np.zeros((input_vector.shape[0],1))), axis=1)
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

    def backprop(self,output,expected, update=True):
        """Performs backpropagation through the neural network for one entry, using squared error"""
        error = expected-output # gradient of the quadratic cost function
        for layer in self.layers[::-1]:
            delta = np.multiply(error, tanh_(layer.output))
            error = np.dot(delta, layer.weights.T)[:,:-1]
            if update:
                layer.weights += np.dot(layer.input.T, delta) * LEARNING_RATE
        return error

    def train(self, examples, labels, epochs):
        """Takes a 2D array of floats as input, formatted with last item as labels and all others as attributes"""
        for x in range(epochs):
            forward = self.feed_forward(examples)
            self.backprop(forward, labels)

# helper functions
def draw_cell(x, y, val):
    """Takes x and y value, and plots a pixel using pyplot"""
    x_arr = [x-0.5, x-0.5, x+0.5, x+0.5]
    y_arr = [y-0.5, y+0.5, y+0.5, y-0.5]
    plt.fill(x_arr, y_arr, color=(0,(val+1.0)/2.0,0))

def draw(arr, window=False):
    """Takes a 1D numpy array and draws a 28x28 image with thresholding"""
    if window:
        for loc, pixel in enumerate(arr):
            draw_cell(loc%IMG_SIZE, -1*loc/IMG_SIZE, pixel)
        plt.show()
    else:
        for row in xrange(IMG_SIZE):
            logging.error("".join(["#" if x>0 else " " for x in arr[row*IMG_SIZE:row*IMG_SIZE+IMG_SIZE]]))

if __name__ == "__main__":
    # Do 20% testing and 80% training by default
    # get test data and train
    results = []
    g_net = NN_Network(g_layers)
    d_net = NN_Network(d_layers)
    for k in xrange(EPOCHS / REPORT):
        logging.error("Beginning epoch \033[93m%s\033[0m" % (k * REPORT))
        """# train from training data
        offset = random.randint(0, 1000-TRAIN_BATCH_SIZE)
        train_input = load_mnist_data('trainingimages', offset=offset, batch_max=TRAIN_BATCH_SIZE)['data']
        train_label = load_mnist_labels('traininglabels', offset=offset, batch_max=TRAIN_BATCH_SIZE)['data'] # get label
        train_label = np.array([[x==y[0] for x in xrange(10)] for y in train_label])
        d_net.train(train_input, train_label, REPORT)
        logging.error("    --finished training from real data")
        # train d from g data
        seed = np.array([[x==y for x in xrange(10)] for y in np.random.randint(0, 10, (TRAIN_BATCH_SIZE))])
        images = g_net.feed_forward(seed) # generate images from seed
        d_net.train(images, np.zeros((TRAIN_BATCH_SIZE, d_layers[-1])), REPORT) # train, marking generator as fake
        logging.error("    --finished training d from g data")"""
        # train g from d data
        train_input = load_mnist_data('trainingimages', offset=1, batch_max=TRAIN_BATCH_SIZE)['data']
        train_label = load_mnist_labels('traininglabels', offset=1, batch_max=TRAIN_BATCH_SIZE)['data']
        seed = np.array([[x==y[0] for x in xrange(10)] for y in train_label])
        goals = train_input
        images = g_net.feed_forward(seed) # generate images from seed
        #d_class = d_net.feed_forward(images)# get classification from discriminator
        #d_grad = d_net.backprop(d_class, seed, update=False) # get gradient from discriminator
        for _ in xrange(REPORT): # backprop result through generator 
            #g_net.backprop(images, images+d_grad)
            g_net.backprop(images, goals)
        logging.error("    --finished training g from d gradient")
        # test
        test_input = load_mnist_data('testimages', offset=0, batch_max=TEST_BATCH_SIZE)['data']
        test_label = load_mnist_labels('testlabels', offset=0, batch_max=TEST_BATCH_SIZE)['data']
        test_sum = 0
        labels = d_net.feed_forward(test_input)
        for index in xrange(len(test_label)):
            test_sum += (max(enumerate(labels[index]), key=lambda b:b[1])[0] == test_label[index])
        logging.error("    --finished testing: accuracy %s" % (float(test_sum)/TEST_BATCH_SIZE))
        results.append(float(test_sum)/TEST_BATCH_SIZE)
    for digit in xrange(10):
        seed = np.array([[x==digit for x in xrange(10)]])
        logging.error("Drawing digit %s" % digit)
        draw(g_net.feed_forward(seed)[0], window=True)
    plt.plot(results)
    plt.show()
