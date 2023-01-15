import unittest
import torch
import nn
import torch.nn.functional as F


class TestActivations(unittest.TestCase):
    def test_tanh(self): self.helper_function(nn.Tanh(), torch.tanh)
        
    def test_sigmoid(self): self.helper_function(nn.Sigmoid(), torch.sigmoid)

    def test_relu(self): self.helper_function(nn.ReLU(), torch.relu)

    def test_leaky_relu(self): self.helper_function(nn.LeakyReLU(), F.leaky_relu)

    def test_GELU(self): self.helper_function(nn.GELU(), F.gelu)

    def test_softmax(self): self.helper_function(nn.Softmax(dim=1), F.softmax)

    def test_relu6(self): self.helper_function(nn.ReLU6(), F.relu6)

    def test_elu(self): self.helper_function(nn.ELU(), F.elu)

    def test_swish(self): self.helper_function(nn.Swish(), F.silu)

    def test_softplus(self): self.helper_function(nn.Softplus(), F.softplus)

    def test_mish(self): self.helper_function(nn.Mish(), F.mish)

    def test_hardshrink(self): self.helper_function(nn.HardShrink(), F.hardshrink)

    @staticmethod
    def helper_function(act: torch.Tensor, torch_act: torch.Tensor) -> None:
        x = torch.randn(100, 100)
        
        torch.allclose(torch_act(x), act(x))

if __name__ == '__main__':
    torch.manual_seed(1337)
    unittest.main()
