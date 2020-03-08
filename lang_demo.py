import numpy as np
import matplotlib.pyplot as plt
import neural, init
import pandas as pd

# Simple cost functions. Working on implementing a logarithmic version, but it isn't working right now
def cost(actualOutput, optimalOutput):
	return np.square(optimalOutput - actualOutput)

def costDiff(actualOutput, optimalOutput):
	return 2 * (optimalOutput - actualOutput)

# Activation functions
def RELU(z):
	return np.where(z > 0, z, 0)

def RELU_diff(z):
	return np.where(z > 0, 1, 0)

def get_datasets(file, training_part):
	"""Retrieves dataset and returns training and test set."""
	wordData = pd.read_csv(file + '.csv')

	train_set = wordData.sample(frac=training_part)
	test_set = wordData.drop(train_set.index)

	return (train_set, test_set)

def test(test_set, net):
	"""Tests network on test set"""
	inputLayer = test_set.to_numpy()[:,3:].astype('float32')
	optimalOutput = test_set.to_numpy()[:,0:2].astype('float32')
	net.feedForward(inputLayer)
	#TODO : Fjerne optimalOutput == 1,1. Håndtere de ordene som er både norske og engelske annerledes
	errorSame = np.where(np.mean(optimalOutput, axis=1) == 1, 1, 0)
	error = np.where(np.argmax(optimalOutput, axis=1) == np.argmax(net.network[-1].a, axis=1), 0, 1)
	print("Feil: {}. Totalt i testsettet: {}. Feilprosent: {}".format(np.sum(errorSame), test_set.shape[0], np.sum(error - errorSame)/test_set.shape[0])*100)

def train(train_set, net, mini_batch_size):
	"""Trains network on training set"""
	error = []
	temp_set = train_set
	for e in range(1):
		train_set = temp_set
		for i in range(len(train_set.index)//mini_batch_size):
			mini_batch = train_set.sample(n = mini_batch_size)
			inputLayer = mini_batch.to_numpy()[:,3:].astype('float32')
			optimalOutput = mini_batch.to_numpy()[:,0:2].astype('float32')
			net.feedForward(inputLayer)
			#print(optimalOutput[0,0], net.network[-1].neurons[0,0])
			net.backPropagate(optimalOutput)
			error.append(np.mean(cost(net.network[-1].a, optimalOutput)))
	plt.plot(error)
	plt.show()

if __name__ == '__main__':

	# Gets dataset
	train_set, test_set = get_datasets('complete', .9)

	# Makes neural network with input layer, two hidden layers and one output layer
	net = neural.Network(learningRate = 0.008)
	net.addLayer(neural.Input(30))
	net.addLayer(neural.Dense(10, activation=RELU, diff_activation=RELU_diff))
	net.addLayer(neural.Dense(10, activation=RELU, diff_activation=RELU_diff))
	net.addLayer(neural.Output(2, costDiff=costDiff))
	net.finalise()

	# Trains and then tests
	train(train_set, net, 64)
	test(test_set, net)

	# Lets user test network with own words. Prints out score and guess
	run = True
	T = ['engelsk', 'norsk']
	while run:
		word = input("Ord: ")
		if word == 'stopp':
			run = False
		inputLayer = init.encode(word)
		net.feedForward(inputLayer)
		print("Nettet tror {} er et {} ord. ({})".format(word, T[np.argmax(net.network[-1].a)], net.network[-1].a))

	# Prints out all weights and biases
	print(net)


