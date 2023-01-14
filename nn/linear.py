import torch
from typing import List
from .containers import Module


class Linear(Module):
  r"""
  Linear layer with Kaiming initialization

  Args:
    - fan_in (int): number of input features
    - fan_out (int): number of output features
    - bias (bool): if ``True``, adds a learnable bias to the output. Default: ``True

  Notes:
    - Kaiming initialization: https://arxiv.org/abs/1502.01852
  """
  def __init__(self, fan_in: int, fan_out: int, bias: bool=True) -> None:
    self.weight = torch.randn((fan_in, fan_out)) / fan_in**0.5 
    self.bias = torch.zeros(fan_out) if bias else None
  
  def forward(self, x : torch.Tensor) -> torch.Tensor:
    self.out = x @ self.weight
    
    if self.bias is not None:
      self.out += self.bias
    
    return self.out
  
  def parameters(self) -> List:
    return [self.weight] + ([] if self.bias is None else [self.bias])



