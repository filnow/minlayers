import numpy as np
import unittest
import torch
import nn


def helper_function(exp_act, act):
        # Compare to troch
        x_torch = torch.randn(3, 3)
        expected_torch = exp_act(x_torch)
        np.testing.assert_array_almost_equal(expected_torch, act(x_torch), decimal=5)


class TestActivations(unittest.TestCase):
    def test_tanh(self):
        act = nn.Tanh()

        exp_act = torch.tanh

        helper_function(exp_act, act)

    def test_sigmoid(self):
        act = nn.Sigmoid()

        exp_act = torch.sigmoid

        helper_function(exp_act, act)

    def test_relu(self):
        act = nn.ReLU()

        exp_act = torch.relu

        helper_function(exp_act, act)


if __name__ == '__main__':
    np.random.seed(1337)
    unittest.main()
