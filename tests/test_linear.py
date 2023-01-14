import unittest
import torch
import nn
import torch.nn.functional as F

class TestLinear(unittest.TestCase):
    def test_linear(self):
        x = torch.randn(100, 100)
        fc = nn.Linear(100, 100)
        torch.allclose(x @ fc.weight, fc(x))

    def test_linear_bias(self):
        x = torch.randn(100, 100)
        fc = nn.Linear(100, 100, bias=True)
        torch.allclose(x @ fc.weight + fc.bias, fc(x))


if __name__ == '__main__':
    torch.manual_seed(1337)
    unittest.main()
