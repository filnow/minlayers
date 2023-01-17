import torch
from typing import List
from .containers import Module


class Linear(Module):
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

  def __repr__(self) -> str:
    return f"Linear(fan_in={self.weight.shape[0]}, fan_out={self.weight.shape[1]}, bias={self.bias is not None})"


