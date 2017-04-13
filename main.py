from loader import load_mnist_data, load_mnist_labels
import numpy as np
import sys

LEARNING_RATE = 0.1
BATCH_SIZE = 1000
EPOCHS = 10000

train_input = load_mnist_data('trainingimages', batch_max=BATCH_SIZE)['data']
train_label = load_mnist_labels('traininglabels', batch_max=BATCH_SIZE)['data']

class Perceptron:
    
    def __init__(self, input_size, output_size):
        self.weight = np.random.uniform(-0.1, 0.1, (input_size, output_size))

    def feedforward(self, input_vector):
        return np.dot(input_vector, self.weight)

    def train(self, input_set, label_set, iter=1):
        for _ in xrange(iter):
            error = label_set - self.feedforward(input_set)
            return
            self.weight += LEARNING_RATE * np.dot(input_set.T, error)
        
perceptrons = []
for label in xrange(10):
    net = Perceptron(784, 1)
    filtered_label = train_label==label
    net.train(train_input, filtered_label, iter=EPOCHS)
    perceptrons.append(net)

offset = int(sys.argv[1])
test_input = load_mnist_data('trainingimages', offset=offset, batch_max=1)['data']
test_label = load_mnist_labels('traininglabels', offset=offset, batch_max=1)['data']

for row in range(28):
    out = ""
    for col in range(28):
        if test_input[0][28*row+col] == 0:
            out += " "
        elif test_input[0][28*row+col] == 0.5:
            out += "+"
        else:
            out += "#"
    print out

print max(enumerate([p.feedforward(test_input)[0][0] for p in perceptrons]), key=lambda (a,b):b)[0]