'''
Problem source:
https://www.deep-ml.com/problem/Single%20Neuron%20with%20Backpropagation

YouTube video:
https://youtu.be/LPfsTFcFqU4
'''

import numpy as np
from questions.q25_single_neuron_with_backpropogation import train_neuron

def test_train_neuron_case1():
    # Test case 1
    weights, bias, mse_values = train_neuron(
        np.array([[1.0, 2.0], [2.0, 1.0], [-1.0, -2.0]]),
        np.array([1, 0, 0]),
        np.array([0.1, -0.2]),
        0.0,
        0.1,
        2
    )
    
    # Check results with tolerances for numerical differences
    assert np.allclose(weights, [0.1036, -0.1425], rtol=0.01), weights
    assert abs(bias - (-0.0167)) < 0.01, bias
    assert np.allclose(mse_values, [0.3033, 0.2942], rtol=0.01), mse_values

def test_train_neuron_case2():
    # Test case 2
    weights, bias, mse_values = train_neuron(
        np.array([[1, 2], [2, 3], [3, 1]]),
        np.array([1, 0, 1]),
        np.array([0.5, -0.2]),
        0,
        0.1,
        3
    )
    # Check results with tolerances for numerical differences
    assert np.allclose(weights, [0.4892, -0.2301], rtol=0.01), weights
    assert abs(bias - 0.0029) < 0.01, bias
    assert np.allclose(mse_values, [0.21, 0.2087, 0.2076], rtol=0.01), mse_values