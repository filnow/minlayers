import numpy as np
import unittest
import torch
from nn.activations import *

#TODO: simplify this

class TestActivations(unittest.TestCase):
    def test_tanh(self):
        activation = Tanh()
        x = np.random.randn(3, 3)
        expected = np.tanh(x)
        np.testing.assert_array_almost_equal(expected, activation(x), decimal=5)
        np.testing.assert_array_equal([], activation.parameters())

        # Compare to troch
        x_torch = torch.from_numpy(x)
        expected_torch = torch.tanh(x_torch)
        np.testing.assert_array_almost_equal(expected_torch.numpy(), activation(x), decimal=5)

    def test_sigmoid(self):
        activation = Sigmoid()
        x = np.random.randn(3, 3)
        expected = (1 + np.exp(-x))**-1
        np.testing.assert_array_almost_equal(expected, activation(x), decimal=5)
        np.testing.assert_array_equal([], activation.parameters())

        # Compare to troch
        x_torch = torch.from_numpy(x)
        expected_torch = torch.sigmoid(x_torch)
        np.testing.assert_array_almost_equal(expected_torch.numpy(), activation(x), decimal=5)
        
    def test_relu(self):
        activation = ReLU()
        x = np.random.randn(3, 3)
        expected = np.maximum(0, x)
        np.testing.assert_array_equal(expected, activation(x))
        np.testing.assert_array_equal([], activation.parameters())

        # Compare to troch
        x_torch = torch.from_numpy(x)
        expected_torch = torch.relu(x_torch)
        np.testing.assert_array_equal(expected_torch.numpy(), activation(x))

if __name__ == '__main__':
    np.random.seed(1337)
    unittest.main()
