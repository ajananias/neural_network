import numpy as np
import scipy.special
# neural network class definition and its methods
class neuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        # weigth matrices initialization with weights sampled from a normal distribution centered around 0 and standard deviation of 1/sqrt(number of incoming links)
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes)) #W_ih
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes)) #W_ho transposed
        # activation function: sigmoid
        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        # signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        # signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)
        # Update the weights between input, hidden and output layers
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
    
    def query(self, inputs_list):
        # input conversion into a transposed 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        # signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        # signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs