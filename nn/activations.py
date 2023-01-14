import torch
from .containers import Module

class Tanh(Module):
    """
    Tanh activation function

    Args:
        x (torch.Tensor): input to the activation function
    
    Returns:
        torch.Tensor: output of the activation function

    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.out = torch.tanh(x)
        return self.out
    


class Sigmoid(Module):
    """
    Sigmoid activation function

    Args:
        x (torch.Tensor): input to the activation function
    
    Returns:
        torch.Tensor: output of the activation function

    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.out = (1 + torch.exp(-x))**-1
        return self.out
    
  

class ReLU(Module):
    """
    ReLU activation function

    Args:
        x (torch.Tensor): input to the activation function
    
    Returns:
        torch.Tensor: output of the activation function

    """ 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.out = torch.relu(x)
        return self.out
    