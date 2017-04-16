from loader import load_mnist_data, load_mnist_labels
from random import randint
from matplotlib import pyplot
import logging
import numpy as np
import sys

LEARNING_RATE = 0.01
BATCH_SIZE = 10
EPOCHS = 19000
TEST_BATCH_SIZE = 20
REPORT = 600
REGULARIZATION_STRENGTH = 0.0

class Perceptron:
    
    def __init__(self, input_size, output_size):
        self.weight = np.random.uniform(-0.1, 0.1, (input_size, output_size))

    def feedforward(self, input_vector):
        return np.dot(input_vector, self.weight)

    def train(self, input_set, label_set, iter=1):
        for _ in xrange(iter):
            error = label_set - self.feedforward(input_set) # pure error
            self.weight += LEARNING_RATE * np.dot(input_set.T, error)
perceptrons = []
for label in xrange(10):
    net = Perceptron(262, 1)
    perceptrons.append(net)

correct = []
for step in xrange(0,EPOCHS,REPORT):
    for p in enumerate(perceptrons):
        offset = randint(0,500)
        train_input = load_mnist_data('trainingimages', offset=offset, batch_max=BATCH_SIZE)['data']
        train_label = load_mnist_labels('traininglabels', offset=offset, batch_max=BATCH_SIZE)['data']
        filtered_label = train_label==p[0]
        p[1].train(train_input, filtered_label, iter=REPORT)
    logging.error(np.dot(np.ones((1,262)), perceptrons[0].weight))
    offset = 0
    test_input = load_mnist_data('testimages', offset=offset, batch_max=TEST_BATCH_SIZE)['data']
    test_label = load_mnist_labels('testlabels', offset=offset, batch_max=TEST_BATCH_SIZE)['data']
    test_sum = 0
    for test_item in xrange(TEST_BATCH_SIZE):
        labels = np.stack([p.feedforward(test_input) for p in perceptrons], axis=2)[test_item][0]
        labels_sorted = [label for label in sorted(enumerate(labels), key=lambda a:a[1], reverse=True)[:3]]
        correct_label = int(test_label[test_item][0])
        test_sum += (labels_sorted[0][0]==correct_label) + 0.0 * (labels_sorted[1][0]==correct_label) + 0.0 * (labels_sorted[2][0]==correct_label)
    correct.append(test_sum)
    logging.warning("Finished step: "+str(step))
    LEARNING_RATE *= 0.7
pyplot.plot(correct)
pyplot.show()