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

    def test_softmax(self): self.helper_function(nn.Softmax(dim=0), F.softmax)

    def test_relu6(self): self.helper_function(nn.ReLU6(), F.relu6)

    def test_elu(self): self.helper_function(nn.ELU(), F.elu)

    def test_swish(self): self.helper_function(nn.Swish(), F.silu)

    def test_softplus(self): self.helper_function(nn.Softplus(), F.softplus)

    def test_mish(self): self.helper_function(nn.Mish(), F.mish)
    
    @staticmethod
    def helper_function(act: torch.Tensor, torch_act: torch.Tensor) -> None:
        x = torch.randn(3, 3)
        
        torch.allclose(torch_act(x), act(x), rtol=1e-5, atol=1e-8)

if __name__ == '__main__':
    unittest.main()
