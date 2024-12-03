'''
Problem source:
https://www.deep-ml.com/problem/Single%20Neuron%20with%20Backpropagation

YouTube video:
https://youtu.be/LPfsTFcFqU4
'''

import numpy as np

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def train_neuron(x: np.ndarray, labels: np.ndarray, weights: np.ndarray, bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
	n = len(labels)
	# we'll label feature dimension as d
	mse_values = []
	for _ in range(epochs):
		# fwd pass
		z = x @ weights + bias # (n,d) @ (d) + (1) -> (n) 
		probabilities = sigmoid(z) # (n)
		mse = np.mean((probabilities - labels)**2) # (1)
		mse_values.append(np.round(mse, 4))
		
		# calc gradients
		dL_dp = (2 / n) * (probabilities - labels) # (1) * ((n) - (n)) -> (n)
		dp_dz = sigmoid(z) * (1 - sigmoid(z)) # (n) -> (n)
		dL_dz = dL_dp * dp_dz # (n) * (n) -> (n)
		dz_db = sum(z) # (n) -> (1)
		dL_db = sum(dL_dz * dz_db) # (1)
		dz_dw = x # (n,d)
		dL_dw = dL_dz @ dz_dw # (n) * (n,d) -> (d)
			
		# gradient step
		weights -= learning_rate * dL_dw
		bias -= learning_rate * dL_db
		
		# keeping sig figs
		weights = np.round(weights, 4)
		bias = round(bias, 4)
	
	return weights.tolist(), bias, mse_values