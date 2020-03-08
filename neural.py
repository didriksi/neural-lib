import numpy as np
import matplotlib.pyplot as plt
import unittest

class TestNeuralNetwork(unittest.TestCase):
	def test_feedforward(self):
		network = Network()
		network.addLayer(Input(1))
		network.addLayer(Dense(1, activation=lambda x: x))
		network.addLayer(Output(1, activation=lambda x: x))
		network.finalise()
		network.network[1].b = np.array([1])
		network.network[1].w = np.array([1])
		network.network[2].b = np.array([2])
		network.network[2].w = np.array([2])
		network.feedForward(np.array([1]))
		self.assertTrue(network.network[2].a == 6)

# Sigmoidal activation functions
def sigmoid(z):
	return 1/(1 + np.exp(-z))

def sigmoidPrime(z):
	return sigmoid(z)*(1-sigmoid(z))

# Simple cost functions. Working on implementing a logarithmic version, but it isn't working right now
def cost(actualOutput, optimalOutput):
	return np.square(optimalOutput - actualOutput)

def costDiff(actualOutput, optimalOutput):
	return 2 * (optimalOutput - actualOutput)

# Different kinds of layers. All are fully connected with previous layer.
class Layer:
	def __init__(self, height):
		self.a = np.zeros(height)
		self.height = height
	def set_prevLayer(self, prevLayerHeight):
		self.w = np.random.normal(0, 1, (prevLayerHeight, self.height))

class Input(Layer):
	def __init__(self, height):
		super().__init__(height)

class Dense(Layer):
	def __init__(self, height, activation=sigmoid, diff_activation=sigmoidPrime):
		super().__init__(height)
		self.z = np.zeros_like(self.a)
		self.d = np.zeros_like(self.a)
		self.b = np.random.normal(0, 1, height)
		self.activation, self.diff_activation = activation, diff_activation
	def feedForward(self, prevLayer):
		self.z = np.dot(prevLayer.a, self.w)+self.b
		self.a = self.activation(self.z)
	def backPropagate(self, nextLayer):
		self.d = np.dot(nextLayer.w, nextLayer.d.T).T * self.diff_activation(self.z)

class Output(Dense):
	def __init__(self, height, activation=sigmoid, diff_activation=sigmoidPrime, costDiff=costDiff):
		super().__init__(height, activation, diff_activation)
		self.costDiff = costDiff
	def backPropagate(self, optimal):
		self.d = self.costDiff(self.a, optimal) * self.diff_activation(self.z)

# Network class that stores layers, and organises interactions between them
class Network:
	def __init__(self,learningRate=0.006):
		self.learningRate = learningRate
		self.network = []
	def addLayer(self,layer):
		self.network.append(layer)
	def finalise(self):
		for i in range(1, len(self.network)):
			self.network[i].set_prevLayer(self.network[i-1].height)
		self.network = np.asarray(self.network)
	def feedForward(self, inputNeurons):
		self.network[0].a = inputNeurons
		for i in range(1,len(self.network)):
			self.network[i].feedForward(self.network[i-1])
	def backPropagate(self, optimalOutput):
		for i in reversed(range(1, len(self.network))):
			if i == len(self.network) - 1:
				backprop_data = optimalOutput
			else:
				backprop_data = self.network[i+1]

			self.network[i].backPropagate(backprop_data)

			self.network[i].b += np.mean(self.network[i].d, axis=0) * self.learningRate
			self.network[i].w += np.dot(self.network[i-1].a.T, self.network[i].d) * self.learningRate
	def __str__(self):
		string = "Neural net\n"
		for i in range(1, len(self.network)):
			string += "Layer {}\nBiases: {}\nWeights: {}\n".format(i, self.network[i].b, self.network[i].w)
		return string


if __name__ == '__main__':
	unittest.main()